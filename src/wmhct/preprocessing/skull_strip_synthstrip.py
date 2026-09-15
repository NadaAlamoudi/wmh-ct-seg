"""
Stage 2.2 - Skull stripping with FreeSurfer SynthStrip (Docker).

Manuscript: Methods, CT preprocessing
Run: wmhct-skull-strip --help

Note: Runs before windowing.
"""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Skull-strip CT volumes with SynthStrip")
    parser.add_argument("-in", "--in_dir", required=True, help="Directory of input NIfTI files")
    parser.add_argument("-out", "--out_dir", required=True, help="Directory for output")
    parser.add_argument("--image", default="freesurfer/synthstrip:latest",
                        help="Docker image tag. Pin to a digest for a reproducible deposit.")
    parser.add_argument("--sudo", action="store_true", help="Prefix the docker call with sudo")
    args = parser.parse_args()

    if args.image.endswith(":latest"):
        print("WARNING: using the :latest SynthStrip tag. Pin it to a digest "
              "(freesurfer/synthstrip@sha256:...) before depositing this repository.")

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    failures = 0
    for file_name in sorted(os.listdir(in_dir)):
        if not file_name.endswith('.nii.gz'):
            continue

        input_file = os.path.join('/input', file_name)
        output_file = os.path.join('/output', file_name.replace('.nii.gz', '_BRAIN.nii.gz'))

        docker_cmd = (['sudo'] if args.sudo else []) + [
            'docker', 'run',
            '-v', f'{in_dir}:/input',
            '-v', f'{out_dir}:/output',
            args.image,
            '-i', input_file,
            '-o', output_file,
        ]

        print('Running command:', ' '.join(docker_cmd))
        result = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            failures += 1
            print(f'Error processing {file_name}:')
            print(result.stderr.decode())

    print(f"Skull stripping complete. Failures: {failures}")


if __name__ == "__main__":
    main()
