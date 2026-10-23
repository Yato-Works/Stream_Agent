"""Remote-backed Transformers backend (Plan.md C7).

This backend never lets ``transformers`` open the weight files itself.  The
model skeleton is built on the ``meta`` device and every parameter is
materialised from bytes that the ``WeightProvider`` fetched through the
bounded cache.

Two materialisation modes are supported:

``lazy``
    Decoder layers are materialised on demand, right before the layer runs,
    and released again afterwards.  The host working set stays proportional to
    one layer instead of the whole model.

``eager``
    Every parameter is materialised up front.  This is the simpler mode and
    the fallback when lazy materialisation is not supported by the model.
"""

from __future__ import annotations

import asyncio
import inspect
import shutil
import tempfile
import warnings
import time
from pathlib import Path
from typing import Any

from ..errors import BackendCapabilityMissing, BackendOutOfMemory
from ..layout.model_map import ExpertPacking
from ..provider import TensorKey
from .base import BackendCapabilities, GenerationResult, LoadContext

METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

_DTYPE_MAP = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    # F8_E8M0 is only available in newer PyTorch releases; when it is
    # missing the backend reports a capability error instead of guessing.
    "F8_E8M0": "float8_e8m0fnu",
}

#: The torch dtype name that corresponds to each safetensors dtype code.  Used
#: to decide whether a cast to the requested compute dtype is actually needed.
_ARTIFACT_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
}


class TransformersBackend:
    """Runs a Transformers causal LM whose weights come from the cache."""

    capabilities = BackendCapabilities(
        backend_id="transformers",
        version="0.1.0",
        architectures=("*",),
        formats=("safetensors",),
        quantizations=("unquantized",),
        devices=("cpu", "cuda"),
        access_granularity=("tensor", "tensor_slice"),
        min_working_memory_bytes=0,
        supports_streaming=False,
        notes=(
            "Weights are materialised from the bounded cache; the loader never "
            "downloads a complete weight file."
        ),
    )

    def __init__(self) -> None:
        self._handle: dict[str, Any] | None = None

    @staticmethod
    def _torch():
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise BackendCapabilityMissing(
                "the transformers backend requires PyTorch to be installed"
            ) from exc
        return torch

    async def _materialize_dir(self, context: LoadContext) -> Path:
        """Fetch the small metadata files into a local directory."""
        target = Path(tempfile.mkdtemp(prefix=f"stream-agent-{context.alias}-", suffix="-meta"))
        layout = context.layout
        for name in METADATA_FILES:
            size = layout.metadata_files.get(name)
            if size is None:
                continue
            try:
                data = await context.provider.read_file(name, 0, size - 1)
            except Exception:
                continue
            (target / name).write_bytes(data)
        return target

    async def load(self, context: LoadContext) -> dict[str, Any]:
        torch = self._torch()
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise BackendCapabilityMissing(
                "the transformers backend requires the 'transformers' package"
            ) from exc

        meta_dir = await self._materialize_dir(context)
        try:
            config = AutoConfig.from_pretrained(str(meta_dir), trust_remote_code=False)
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(meta_dir), trust_remote_code=False
                )
            except Exception:
                tokenizer = None

            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
            model.eval()
            _restore_non_persistent_buffers(model, config, context.device)

            materializer = _Materializer(
                model=model,
                layout=context.layout,
                provider=context.provider,
                torch=torch,
                device=context.device,
                dtype=context.dtype,
                materialize_mode=str(context.options.get("materialize", "lazy")),
                loop=asyncio.get_running_loop(),
            )
            await materializer.prepare()
            if materializer.mode != "eager":
                materializer.install_lazy_hooks()

            handle = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "materializer": materializer,
                "alias": context.alias,
                "revision": context.revision,
            }
            self._handle = handle
            return handle
        finally:
            shutil.rmtree(meta_dir, ignore_errors=True)

    async def unload(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        materializer = handle.get("materializer")
        if materializer is not None:
            materializer.remove_lazy_hooks()
        model = handle.get("model")
        if model is None:
            return
        try:
            model.to("meta")
        except Exception:  # pragma: no cover - best effort release
            pass

    async def generate(self, handle: Any, prompt: str, **params: Any) -> GenerationResult:
        torch = self._torch()
        model = handle["model"]
        tokenizer = handle["tokenizer"]
        if tokenizer is None:
            raise BackendCapabilityMissing(
                "the tokenizer could not be loaded; token based generation is unavailable"
            )
        max_new_tokens = int(params.get("max_tokens", 32))
        temperature = float(params.get("temperature", 0.0) or 0.0)
        provider = handle["materializer"].provider

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        if temperature > 0:
            kwargs["temperature"] = temperature
            if params.get("top_p"):
                kwargs["top_p"] = float(params["top_p"])

        def _run() -> Any:
            with torch.inference_mode():
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )

        started = time.perf_counter()
        # The weight wait is measured over the generation interval only; the
        # wait that happened while the model was opened is reported
        # separately so that the fraction stays comparable.
        load_weight_wait_ms = provider.reset_wait_timer()
        try:
            output = await asyncio.to_thread(_run)
        except torch.cuda.OutOfMemoryError as exc:
            raise BackendOutOfMemory(
                "the backend ran out of accelerator memory while generating"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise BackendOutOfMemory(str(exc)) from exc
            raise

        total_ms = (time.perf_counter() - started) * 1000.0
        new_tokens = output[0][input_ids.shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated = int(new_tokens.shape[0])
        weight_wait_ms = provider.reset_wait_timer()
        return GenerationResult(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=generated,
            ttft_ms=total_ms if generated else 0.0,
            generation_ms=total_ms,
            weight_wait_ms=weight_wait_ms,
            finish_reason="length" if generated >= max_new_tokens else "stop",
            load_weight_wait_ms=load_weight_wait_ms,
        )

    def expert_packing(self) -> dict[str, ExpertPacking]:
        return {}


def _restore_non_persistent_buffers(model: Any, config: Any, device: str) -> None:
    """Recompute non persistent buffers (rotary frequencies) on a real device.

    Building the skeleton on the ``meta`` device also moves buffers such as
    ``inv_freq`` onto the meta device.  Those values are derived from the
    configuration, so the owning module is re-created on the target device.
    """
    torch = import_torch()
    for module in model.modules():
        buffers = dict(module._buffers)
        if not buffers:
            continue
        if any(param is not None for param in module._parameters.values()):
            # Only modules without parameters are safe to re-create.
            continue
        try:
            signature = inspect.signature(type(module).__init__)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        parameters = signature.parameters
        if "config" not in parameters:
            continue
        module_config = getattr(module, "config", config)
        rebuilt = _rebuild_module(type(module), module_config, device)
        if rebuilt is None:
            continue
        for name, buffer in rebuilt._buffers.items():
            module._buffers[name] = buffer.to(device) if torch.is_tensor(buffer) else buffer


def _rebuild_module(module_type, module_config, device):
    """Re-create a module so that its derived buffers become real tensors.

    Different model families accept the configuration differently: some take
    ``(config)``, some ``(config, device)`` and some only ``(config=...)``.
    Every known convention is attempted, and ``None`` is returned when the
    module cannot be re-created at all.
    """
    attempts = (
        lambda: module_type(module_config, device=device),
        lambda: module_type(module_config),
        lambda: module_type(config=module_config, device=device),
        lambda: module_type(config=module_config),
    )
    for attempt in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return attempt()
        except Exception:
            continue
    return None


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise BackendCapabilityMissing(
            "the transformers backend requires PyTorch to be installed"
        ) from exc
    return torch


class _Materializer:
    """Fills model parameters from the cache, lazily when possible."""

    def __init__(
        self,
        *,
        model: Any,
        layout: Any,
        provider: Any,
        torch: Any,
        device: str,
        dtype: str,
        materialize_mode: str = "lazy",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.model = model
        self.layout = layout
        self.provider = provider
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.mode = materialize_mode
        self.loop = loop
        self.materialized_bytes = 0
        self._groups: dict[int, list[str]] = {}
        self._meta_params: dict[str, Any] = {}
        self._layer_names: set[str] = set()
        self._handles: list[Any] = []

    # -- helpers -----------------------------------------------------------
    def _resolve_name(self, name: str) -> str | None:
        layout = self.layout
        if layout.has_tensor(name):
            return name
        mapping = getattr(self.model, "_checkpoint_conversion_mapping", None) or {}
        for prefix, replacement in mapping.items():
            if name.startswith(replacement):
                candidate = prefix + name[len(replacement) :]
                if layout.has_tensor(candidate):
                    return candidate
            if name.startswith(prefix):
                candidate = replacement + name[len(prefix) :]
                if layout.has_tensor(candidate):
                    return candidate
        for prefix in ("model.", "transformer.", "gpt_neox."):
            if name.startswith(prefix):
                stripped = name[len(prefix) :]
                for candidate in (stripped, f"model.{stripped}", f"transformer.{stripped}"):
                    if layout.has_tensor(candidate):
                        return candidate
        return None

    def _module_for(self, full_name: str) -> tuple[Any, str]:
        parts = full_name.split(".")
        module = self.model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module, parts[-1]

    def _assign(self, full_name: str, tensor: Any) -> None:
        module, name = self._module_for(full_name)
        old = module._parameters.get(name)
        requires_grad = bool(getattr(old, "requires_grad", False))
        module._parameters[name] = self.torch.nn.Parameter(
            tensor, requires_grad=requires_grad
        )

    def _install_parameter(self, full_name: str, parameter: Any) -> None:
        """Install an already built parameter object under ``full_name``."""
        module, name = self._module_for(full_name)
        module._parameters[name] = parameter

    def _restore_meta(self, full_name: str) -> None:
        original = self._meta_params.get(full_name)
        if original is None:
            return
        module, name = self._module_for(full_name)
        module._parameters[name] = original

    # -- materialisation ---------------------------------------------------
    def _torch_dtype(self, dtype: str):
        name = _DTYPE_MAP.get(dtype.upper())
        if name is None:
            return None
        resolved = getattr(self.torch, name, None)
        if resolved is None:
            raise BackendCapabilityMissing(
                f"the installed PyTorch has no dtype for {dtype!r}; "
                "the artifact cannot be materialised on this machine",
                details={"dtype": dtype, "torch_dtype": name},
            )
        return resolved

    async def _materialize_names(self, names: list[str]) -> None:
        """Materialise a group of parameter names.

        The names may belong to a tied group: several aliases can share one
        artifact tensor (for example ``model.embed_tokens.weight`` and
        ``lm_head.weight``).  The artifact stores such a tensor only once, so it
        is fetched once and installed for every alias of the group.  Replacing
        only one alias would leave the other one on the meta device, and the
        model would then emit plausible but completely wrong predictions.
        """
        resolved_of: dict[str, str | None] = {
            name: self._resolve_name(name) for name in names
        }
        wanted = sorted({r for r in resolved_of.values() if r is not None})
        if not wanted:
            return

        built: dict[str, Any] = {}  # artifact tensor -> torch tensor
        for resolved in wanted:
            buffer = await self.provider.read_tensor(TensorKey(tensor=resolved))
            source_dtype = self._torch_dtype(buffer.dtype)
            if source_dtype is None:
                raise BackendCapabilityMissing(
                    f"the artifact stores {buffer.dtype!r}, which this PyTorch build "
                    "cannot interpret",
                    details={"dtype": buffer.dtype, "tensor": resolved},
                )
            raw = self.torch.frombuffer(bytearray(buffer.data), dtype=self.torch.uint8)
            try:
                view = raw.view(source_dtype)
            except RuntimeError as exc:  # pragma: no cover - alignment problems
                raise BackendCapabilityMissing(
                    f"cannot reinterpret {buffer.dtype} bytes as {source_dtype}: {exc}"
                ) from exc

            shape: tuple[int, ...] = ()
            for alias, alias_resolved in resolved_of.items():
                if alias_resolved != resolved:
                    continue
                module, name = self._module_for(alias)
                param = module._parameters.get(name)
                if param is not None:
                    shape = tuple(param.shape)
                    break

            expected_elements = 1
            for dim in shape:
                expected_elements *= dim
            if shape and view.numel() != expected_elements:
                raise BackendCapabilityMissing(
                    f"the fetched range holds {view.numel()} {buffer.dtype} elements but "
                    f"the parameter needs {expected_elements}",
                    details={
                        "tensor": resolved,
                        "shape": list(shape),
                        "elements_fetched": int(view.numel()),
                        "elements_expected": expected_elements,
                        "bytes_fetched": len(buffer.data),
                    },
                )

            target_dtype = source_dtype
            if (
                self.dtype in ("float16", "bfloat16", "float32")
                and buffer.dtype in ("F32", "F16", "BF16")
                and self.dtype != _ARTIFACT_DTYPE_NAMES.get(buffer.dtype)
            ):
                target_dtype = getattr(self.torch, self.dtype)

            try:
                tensor = view.reshape(shape) if shape else view
            except RuntimeError as exc:
                raise BackendCapabilityMissing(
                    f"the fetched range does not match the expected shape: {exc}",
                    details={
                        "tensor": resolved,
                        "shape": list(shape),
                        "elements_fetched": int(view.numel()),
                    },
                ) from exc
            try:
                tensor = tensor.to(dtype=target_dtype).to(self.device)
            except Exception as exc:
                raise BackendOutOfMemory(
                    f"not enough memory to materialize {resolved}: {exc}"
                ) from exc
            built[resolved] = tensor
            self.materialized_bytes += len(buffer.data)

        # Aliases that the artifact stores under their own name.
        for alias, alias_resolved in resolved_of.items():
            if alias_resolved is None:
                continue
            module, name = self._module_for(alias)
            old = module._parameters.get(name)
            requires_grad = bool(getattr(old, "requires_grad", False))
            module._parameters[name] = self.torch.nn.Parameter(
                built[alias_resolved], requires_grad=requires_grad
            )

        # Aliases that the artifact does not store separately (tied weights)
        # reuse the tensor of the alias whose shape matches them.
        for alias, alias_resolved in resolved_of.items():
            if alias_resolved is not None:
                continue
            module, name = self._module_for(alias)
            param = module._parameters.get(name)
            wanted_shape = tuple(param.shape) if param is not None else ()
            candidates = [
                tensor
                for tensor in built.values()
                if not wanted_shape or tuple(tensor.shape) == wanted_shape
            ]
            if not candidates:
                continue
            old = module._parameters.get(name)
            requires_grad = bool(getattr(old, "requires_grad", False))
            module._parameters[name] = self.torch.nn.Parameter(
                candidates[0], requires_grad=requires_grad
            )

    # -- preparation -------------------------------------------------------
    async def prepare(self) -> None:
        parameters = list(self.model.named_parameters(recurse=True, remove_duplicate=False))
        for full_name, param in parameters:
            self._meta_params.setdefault(full_name, param)
            self._groups.setdefault(id(param), []).append(full_name)

        layer_names = self._layer_parameter_names()
        self._layer_names = set(layer_names)

        missing: list[str] = []
        for group in self._groups.values():
            if self.mode != "eager" and any(name in layer_names for name in group):
                continue  # materialised lazily, right before the layer runs
            resolved = self._resolve_name(group[0])
            if resolved is None:
                # A tied weight may legitimately be absent from the artifact.
                if any(self._resolve_name(name) for name in group):
                    missing.append(group[0])
                continue
            await self._materialize_names(list(group))
        if missing:
            raise BackendCapabilityMissing(
                "the artifact does not contain every parameter the model needs",
                details={"missing_parameters": missing[:20], "count": len(missing)},
            )

    def _layer_parameter_names(self) -> set[str]:
        """Full names of the parameters that belong to the decoder layers."""
        container = self._find_layer_container()
        if container is None:
            return set()
        names: set[str] = set()
        for prefix, module in container:
            for name, _ in module.named_parameters(recurse=True):
                names.add(f"{prefix}{name}")
        return names

    def _find_layer_container(self) -> list[tuple[str, Any]] | None:
        torch = self.torch
        best: list[tuple[str, Any]] | None = None
        best_len = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > best_len:
                best_len = len(module)
                prefix = ""
                for name, candidate in self.model.named_modules():
                    if candidate is module:
                        prefix = f"{name}." if name else ""
                        break
                best = [(f"{prefix}{index}.", layer) for index, layer in enumerate(module)]
        return best

    def _layer_modules(self) -> list[tuple[str, Any]]:
        return self._find_layer_container() or []

    # -- lazy hooks --------------------------------------------------------
    def install_lazy_hooks(self) -> None:
        """Materialise a decoder layer right before it runs and release it after."""
        for _, layer in self._layer_modules():
            self._handles.append(layer.register_forward_pre_hook(self._pre_hook))
            self._handles.append(layer.register_forward_hook(self._post_hook))

    def remove_lazy_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, module: Any, args: Any):  # pragma: no cover - torch callback
        names = self._names_of_module(module)
        coro = self._materialize_names(names)
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        else:  # pragma: no cover - fallback for synchronous use
            asyncio.get_event_loop().run_until_complete(coro)
        return None

    def _post_hook(self, module: Any, args: Any, output: Any):  # pragma: no cover
        for name in self._names_of_module(module):
            self._restore_meta(name)
        return None

    def _names_of_module(self, module: Any) -> list[str]:
        for prefix, layer in self._layer_modules():
            if layer is module:
                return [
                    f"{prefix}{name}"
                    for name, _ in module.named_parameters(recurse=True)
                ]
        return []


__all__ = ["TransformersBackend"]
