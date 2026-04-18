#!/usr/bin/env python3
import os
import sys

import audio_mixer


def main() -> int:
    output_dir = os.path.join(os.path.dirname(__file__), 'static', 'mp3', 'soundfont_previews')
    os.makedirs(output_dir, exist_ok=True)

    available = audio_mixer.list_available_soundfonts()
    if not available:
        print('No soundfonts found.')
        return 1

    for soundfont_path in available:
        base = os.path.splitext(os.path.basename(soundfont_path))[0]
        rel = os.path.relpath(soundfont_path, os.path.dirname(__file__))
        label = rel.replace(os.sep, '_').replace('.', '_')
        output_mp3 = os.path.join(output_dir, f'{label}_{base}_preview.mp3')
        audio_mixer.render_soundfont_preview(soundfont_path, output_mp3)
        print(f'{soundfont_path} -> {output_mp3}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())