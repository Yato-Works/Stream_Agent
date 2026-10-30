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
from ..prefetch import PrefetchScheduler
from .base import BackendCapabilities, GenerationResult, LoadContext

from .selective_experts import ExpertSpec, SelectiveExpertMaterializer

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
                prefetch_scheduler=(
                    PrefetchScheduler(context.provider, telemetry=context.telemetry)
                    if context.options.get("prefetch_layers", False)
                    else None
                ),
            )
            # Wire selective-MoE configuration from options.  Each spec dict
            # must have keys: tensor, dim, num_experts (proj is optional).
            for spec_dict in context.options.get("expert_specs", []) or []:
                materializer.register_expert_spec(
                    ExpertSpec(
                        tensor=spec_dict["tensor"],
                        dim=int(spec_dict["dim"]),
                        num_experts=int(spec_dict["num_experts"]),
                        proj=spec_dict.get("proj", "unknown"),
                    )
                )
            materializer.router_resolver = context.options.get("router_resolver")
            selective_requested = bool(
                context.options.get("experimental_selective_moe", False)
            )
            if selective_requested:
                # Selective materialisation is opt-in because it can produce
                # silently wrong output when the supplied router resolver does
                # not reproduce the model's own top-k selection exactly.
                if not materializer.expert_specs:
                    raise BackendCapabilityMissing(
                        "experimental_selective_moe requires at least one expert spec"
                    )
                if materializer.router_resolver is None:
                    raise BackendCapabilityMissing(
                        "experimental_selective_moe requires a router_resolver"
                    )
                warnings.warn(
                    "experimental_selective_moe is enabled: unselected experts are "
                    "zero-filled and the backend trusts the supplied router resolver",
                    RuntimeWarning,
                    stacklevel=2,
                )
            materializer.selective_moe_enabled = selective_requested
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
            if materializer.prefetch_scheduler is not None:
                await materializer.prefetch_scheduler.cancel_all()
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
            # The backend calls ``model.generate`` as one blocking operation,
            # so it cannot observe the first token.  Leaving TTFT unset is the
            # honest answer; reporting ``total_ms`` would be total generation
            # time under a different name.
            ttft_ms=None,
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
        prefetch_scheduler: PrefetchScheduler | None = None,
    ) -> None:
        self.model = model
        self.layout = layout
        self.provider = provider
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.mode = materialize_mode
        self.loop = loop
        self.prefetch_scheduler = prefetch_scheduler
        self.materialized_bytes = 0
        self._groups: dict[int, list[str]] = {}
        self._meta_params: dict[str, Any] = {}
        self._layer_names: set[str] = set()
        self._handles: list[Any] = []
        self.layer_profile: dict[str, dict[str, float | int]] = {}
        # -- selective MoE ---------------------------------------------------
        self.expert_specs: dict[str, ExpertSpec] = {}
        self.router_resolver: Any | None = None
        self.selective_moe_enabled = False
        self._expert_full_tensors: dict[str, Any] = {}

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

    # -- selective MoE ------------------------------------------------------
    def register_expert_spec(self, spec: ExpertSpec) -> None:
        """Declare how one architecture packs its experts.

        The spec is also forwarded to the layout so that ``provider.read_expert``
        can resolve the byte slice for each expert id.
        """
        self.expert_specs[spec.tensor] = spec
        self.layout.register_expert_packing(
            ExpertPacking(tensor=spec.tensor, dim=spec.dim, num_experts=spec.num_experts)
        )

    def _param_name_for_artifact(self, artifact_name: str) -> str | None:
        """Find the model parameter name that resolves to ``artifact_name``."""
        for full_name, _ in self.model.named_parameters(recurse=True, remove_duplicate=False):
            if self._resolve_name(full_name) == artifact_name:
                return full_name
        return None

    def _is_expert_layer(self, module: Any) -> bool:
        """Return True if ``module`` contains expert parameters."""
        for name, _ in module.named_parameters(recurse=True):
            if "experts" in name:
                return True
        return False

    def _expert_param_names(self, module: Any) -> list[str]:
        """Return full names of expert parameters owned by ``module``."""
        prefix = ""
        for layer_prefix, layer in self._layer_modules():
            if layer is module:
                prefix = layer_prefix
                break
        return [
            f"{prefix}{name}"
            for name, _ in module.named_parameters(recurse=True)
            if "experts" in name
        ]

    async def _materialize_expert_slices(
        self, layer_module: Any, selected_experts: dict[str, tuple[int, ...]]
    ) -> None:
        """Fetch only the router-selected expert slices and install them.

        ``selected_experts`` maps artifact tensor name -> selected expert ids.
        The full expert parameter tensors are allocated once (zeros) and the
        fetched slices are copied into the corresponding rows.
        """
        materializer = SelectiveExpertMaterializer(
            provider=self.provider, layout=self.layout
        )
        for tensor, spec in self.expert_specs.items():
            materializer.register_spec(spec)

        result = await materializer.materialize(selected_experts)
        if not result.selected:
            return

        torch = self.torch
        source_dtype_map: dict[str, Any] = {}

        for tensor_key, buffer in result.per_tensor.items():
            tensor_name, expert_id_str = tensor_key.rsplit("::", 1)
            expert_id = int(expert_id_str)
            spec = self.expert_specs[tensor_name]

            # Resolve the torch dtype once per artifact dtype.
            if buffer.dtype not in source_dtype_map:
                source_dtype_map[buffer.dtype] = self._torch_dtype(buffer.dtype)
            source_dtype = source_dtype_map[buffer.dtype]

            raw = torch.frombuffer(bytearray(buffer.data), dtype=torch.uint8)
            view = raw.view(source_dtype)

            # Target dtype cast (matches _materialize_names logic).
            target_dtype = source_dtype
            if (
                self.dtype in ("float16", "bfloat16", "float32")
                and buffer.dtype in ("F32", "F16", "BF16")
                and self.dtype != _ARTIFACT_DTYPE_NAMES.get(buffer.dtype)
            ):
                target_dtype = getattr(torch, self.dtype)

            # Full parameter shape and the exact per-expert window the provider
            # used.  Deriving it from the declared packing keeps the copy
            # destination in sync with the fetched byte slice.
            ref = self.layout.tensor(spec.tensor)
            full_shape = tuple(ref.shape)
            packing = self.layout.expert_packing[spec.tensor]
            _, _, per_expert_size = packing.slice_for(expert_id, full_shape)

            param_full_name = self._param_name_for_artifact(tensor_name)
            if param_full_name is None:
                raise BackendCapabilityMissing(
                    f"no model parameter resolves to artifact tensor {tensor_name!r}",
                    details={"tensor": tensor_name},
                )
            module, name = self._module_for(param_full_name)
            param = self._expert_full_tensors.get(tensor_name)
            if param is None or module._parameters.get(name) is not param:
                # Allocate (or re-allocate) the full expert parameter on the
                # target device in the compute dtype.  Reusing a tensor that is
                # no longer installed on the module would write into a detached
                # object.
                full_tensor = torch.zeros(
                    full_shape, dtype=target_dtype, device=self.device
                )
                old = module._parameters.get(name)
                requires_grad = bool(getattr(old, "requires_grad", False))
                param = torch.nn.Parameter(full_tensor, requires_grad=requires_grad)
                module._parameters[name] = param
                self._expert_full_tensors[tensor_name] = param

            # Reshape the fetched slice to the per-expert shape.
            expert_shape = list(full_shape)
            expert_shape[spec.dim] = per_expert_size
            expert_tensor = view.reshape(tuple(expert_shape))

            expert_tensor = expert_tensor.to(dtype=target_dtype).to(self.device)

            # Copy the expert slice into the full parameter tensor.
            with torch.no_grad():
                slices = [slice(None)] * len(full_shape)
                slices[spec.dim] = slice(
                    expert_id * per_expert_size, (expert_id + 1) * per_expert_size
                )
                param.data[tuple(slices)] = expert_tensor

            self.materialized_bytes += len(buffer.data)

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

    def _run_coro(self, coro: Any) -> None:
        """Run a materialisation coroutine from a synchronous torch hook."""
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        else:  # pragma: no cover - fallback for synchronous use
            asyncio.get_event_loop().run_until_complete(coro)

    def _pre_hook(self, module: Any, args: Any):  # pragma: no cover - torch callback
        layer_name = next(
            (prefix.rstrip(".") for prefix, layer in self._layer_modules() if layer is module),
            "unknown_layer",
        )
        started = time.perf_counter()
        bytes_before = self.materialized_bytes
        try:
            return self._pre_hook_impl(module, args)
        finally:
            stats = self.layer_profile.setdefault(
                layer_name, {"calls": 0, "materialize_ms": 0.0, "materialized_bytes": 0}
            )
            stats["calls"] = int(stats["calls"]) + 1
            stats["materialize_ms"] = float(stats["materialize_ms"]) + (
                time.perf_counter() - started
            ) * 1000.0
            stats["materialized_bytes"] = int(stats["materialized_bytes"]) + (
                self.materialized_bytes - bytes_before
            )

    def _pre_hook_impl(self, module: Any, args: Any):
        names = self._names_of_module(module)
        if not names:
            return None
        expert_names = [n for n in names if "experts" in n]
        non_expert_names = [n for n in names if "experts" not in n]

        # Materialise non-expert parameters of the layer normally.
        if non_expert_names:
            self._run_coro(self._materialize_names(non_expert_names))

        if not expert_names:
            return None

        # Selective MoE is opt-in and requires an explicit router resolver.
        # Every other combination falls back to the complete, correct expert
        # bank instead of leaving unselected experts zero-filled.
        if not self.selective_moe_enabled or self.router_resolver is None:
            self._run_coro(self._materialize_names(expert_names))
            return None

        try:
            hidden_states = args[0] if args else None
            if hidden_states is None:
                raise ValueError("the layer did not receive hidden states")
            selected_ids = self.router_resolver(module, hidden_states)
            selected = tuple(sorted({int(expert) for expert in selected_ids}))
        except Exception:
            # A resolver that cannot answer must not silently disable expert
            # loading: fall back to the dense path for this pass.
            self._run_coro(self._materialize_names(expert_names))
            return None

        if not selected:
            self._run_coro(self._materialize_names(expert_names))
            return None

        selected_experts: dict[str, tuple[int, ...]] = {}
        for expert_name in expert_names:
            resolved = self._resolve_name(expert_name)
            if resolved is not None:
                selected_experts[resolved] = selected

        self._run_coro(self._materialize_expert_slices(module, selected_experts))
        return None

    def _post_hook(self, module: Any, args: Any, output: Any):  # pragma: no cover
        # The full expert tensors installed for this layer are about to be
        # replaced by the original meta parameters.  Forget them: reusing a
        # detached tensor on the next pass would leave the module on ``meta``
        # and produce a hard failure (or silent zeros).
        self._expert_full_tensors.clear()
        for name in self._names_of_module(module):
            self._restore_meta(name)
        if self.prefetch_scheduler is not None:
            layers = self._layer_modules()
            current = next((i for i, (_, layer) in enumerate(layers) if layer is module), -1)
            if 0 <= current < len(layers) - 1:
                next_module = layers[current + 1][1]
                names = [
                    resolved
                    for name in self._names_of_module(next_module)
                    if (resolved := self._resolve_name(name)) is not None
                ]
                keys = [TensorKey(tensor=name) for name in dict.fromkeys(names)]
                if keys:
                    self._run_coro(self.prefetch_scheduler.schedule(keys, priority=1))
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
