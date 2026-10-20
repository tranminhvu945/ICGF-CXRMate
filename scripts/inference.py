#!/usr/bin/env python3
"""Run report generation using an ICGF-CXRMate Lightning checkpoint.

No visualization, baseline comparison or clinical data export. Only selected
Findings/Impression text and prompt length are printed.
"""
import argparse
import sys
from pathlib import Path

# Permit `python scripts/inference.py ...` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Stage 2/3 ICGF .ckpt file')
    p.add_argument('--database-dir', type=Path, required=True)
    p.add_argument('--index', type=int, default=0, help='Index in the linked test split')
    p.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--num-tokens', type=int, default=16)
    p.add_argument('--compressor', default='image_conditioned_query',
                   choices=('image_conditioned_query','query','none'))
    p.add_argument('--fusion', default='gated_cross_attention',
                   choices=('gated_cross_attention','concat'))
    p.add_argument('--prompt-mode', default='retain_context',
                   choices=('retain_context','fused_only'))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.database_dir.is_dir():
        raise FileNotFoundError(args.database_dir)

    import torch
    from cxrmate_ed.lightning_module import MIMICIVEDCXRReportGen

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or choose a CUDA host.')
    if args.compressor == 'none' and args.fusion != 'concat':
        raise ValueError('gated fusion requires a patient data compressor')

    lit_model = MIMICIVEDCXRReportGen(
        warm_start_modules=False,
        exp_dir_trial=str(Path('outputs') / 'inference'),
        history=0,
        add_time_deltas=True,
        tables_filter=['mimic_cxr_sectioned', 'triage', 'medrecon'],
        prompt_report_sections_filter=['indication', 'history'],
        database_dir=str(args.database_dir),
        mbatch_size=1,
        decoder_max_len=args.max_length,
        lr=0.0,
        num_test_beams=args.num_beams,
        max_train_images_per_study=5,
        num_workers=0,
        prefetch_factor=2,
        patient_data_compressor=args.compressor,
        patient_data_compressor_num_tokens=args.num_tokens if args.compressor != 'none' else 0,
        patient_data_compressor_num_heads=8,
        patient_data_compressor_dropout=0.0,
        patient_data_compressor_token_type='indication',
        patient_data_fusion=args.fusion,
        patient_data_fusion_num_heads=8,
        patient_data_fusion_dropout=0.0,
        patient_data_fusion_prompt_mode=args.prompt_mode,
    )
    # Only load checkpoints from trusted sources (torch.load can deserialize objects).
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    if any(key.startswith('model.') for key in state):
        state = {key[len('model.'):]: value for key,value in state.items()
                 if key.startswith('model.')}
    elif any(key.startswith('module.') for key in state):
        state = {key[len('module.'):]: value for key,value in state.items()}

    if args.compressor != 'none' and not any('patient_data_compressor' in key for key in state):
        raise ValueError('Checkpoint has no compressor parameters; select the correct Stage 2/3 checkpoint.')
    if args.fusion == 'gated_cross_attention' and not any('patient_data_fusion_module' in key for key in state):
        raise ValueError('Checkpoint has no gated fusion parameters; select an ICGF checkpoint.')

    missing, unexpected = lit_model.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError('Checkpoint is incompatible: missing=' + repr(missing[:15])
                           + ', unexpected=' + repr(unexpected[:15])
                           + '. Check --num-tokens, --compressor and --fusion.')
    lit_model = lit_model.to(args.device).eval()
    model, tokenizer = lit_model.model, lit_model.tokenizer

    test_set = model.get_dataset(database_dir=str(args.database_dir), test_set_only=True)
    if not 0 <= args.index < len(test_set):
        raise IndexError(f'--index {args.index} is outside 0..{len(test_set)-1}')
    sample = test_set[args.index]
    study_id = sample.get('study_id', 'N/A')
    batch = {key: value.to(args.device).unsqueeze(0) if isinstance(value, torch.Tensor)
             else [value] for key,value in sample.items()}

    with torch.no_grad():
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = model.prepare_inputs(
            tokenizer=tokenizer, **batch)
        output_ids = model.generate(
            input_ids=bos_token_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            special_token_ids=[tokenizer.sep_token_id],
            max_length=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=args.num_beams,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']
        findings, impression = model.split_and_decode_sections(
            output_ids, [tokenizer.sep_token_id, tokenizer.eos_token_id], tokenizer)

    print('Study ID:', study_id)
    print('Prompt length:', int(attention_mask.sum(dim=1).item()))
    print('Findings:', findings[0])
    print('Impression:', impression[0])


if __name__ == '__main__':
    main()
