"""Just-for-fun end-to-end generation: prompt → AR codes → FM → wav.

    uv run music3-gen --prompt "dark synthwave, driving bass, 120 bpm" --seconds 30
    uv run music3-gen -p "..." -l "[Verse]\\nneon lights ahead" -s 60 --seed 7
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch

from .generate_codes import generate
from .loading import load_music3_ar, load_tokenizer
from .prompt import encode_prompt
from .synth import FM_CFG_SCALE, FM_STEPS, collect_frame_hiddens, load_synthesis_components, synthesize


def slugify(text: str, max_words: int = 5) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()[:max_words]
    return "-".join(words) or "track"


def main() -> None:
    import soundfile
    from rich.console import Console
    from rich.panel import Panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", "-p", required=True, help="music description (free text or structured caption)")
    parser.add_argument("--lyrics", "-l", default="[Instrumental]", help='lyrics with [section] tags (default: instrumental)')
    parser.add_argument("--seconds", "-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: out/gen/<prompt-slug>-s<seed>.wav")
    parser.add_argument("--steps", type=int, default=FM_STEPS, help="flow-matching steps per chunk")
    parser.add_argument("--cfg", type=float, default=FM_CFG_SCALE, help="flow-matching guidance scale")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-quant", action="store_true", help="bf16 8B instead of NF4 (needs ~30GB VRAM)")
    args = parser.parse_args()

    console = Console()
    out = args.out or Path("out/gen") / f"{slugify(args.prompt)}-s{args.seed}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    tokenizer = load_tokenizer()
    model = load_music3_ar(quantize=not args.no_quant, device=args.device, with_depth=True)
    components = load_synthesis_components(device=args.device)

    prompt_ids = encode_prompt(tokenizer, args.prompt, args.lyrics)
    codes = generate(model, prompt_ids, int(args.seconds * 25), args.seed, args.device)
    hiddens = collect_frame_hiddens(
        model,
        torch.tensor([prompt_ids], device=args.device),
        codes.unsqueeze(0).to(args.device),
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    waveform = synthesize(
        hiddens, *components, generator=generator, num_steps=args.steps, cfg_scale=args.cfg
    )
    soundfile.write(str(out), waveform.squeeze(0).cpu().numpy().T, 44100)

    duration = waveform.shape[-1] / 44100
    console.print(
        Panel.fit(
            f"[bold green]{out}[/bold green]\n"
            f"{duration:.1f}s audio ({codes.shape[0]} frames) in {time.time() - started:.0f}s wall\n"
            f"seed [cyan]{args.seed}[/cyan]  fm steps [cyan]{args.steps}[/cyan]  cfg [cyan]{args.cfg}[/cyan]",
            title="🎵 track ready",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
