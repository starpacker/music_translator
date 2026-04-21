"""
run_reference_erhu.py — Run the reference score_recognition_v4 algorithm
on all 8 pages of 二胡.pdf and combine output into a single PDF.
"""
import os
import sys
import shutil
import fitz  # PyMuPDF
from glob import glob
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# Add reference repo to path
REPO_DIR = os.path.join(os.path.dirname(__file__),
    'repo', 'translate-staff-to-simple-musical-notation-master',
    'score_recognition_v4')
sys.path.insert(0, REPO_DIR)

def prepare_input():
    """Convert 二胡.pdf to images at 500 DPI into repo's input dir."""
    input_dir = os.path.join(REPO_DIR, 'input')
    output_dir = os.path.join(REPO_DIR, 'output')
    output_dir2 = os.path.join(REPO_DIR, 'output_2')

    # Clear directories
    for d in [input_dir, output_dir, output_dir2]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    # Convert PDF pages at 500 DPI
    pdf_path = os.path.join(os.path.dirname(__file__), '二胡.pdf')
    doc = fitz.open(pdf_path)
    print(f"Converting {doc.page_count} pages at 500 DPI...")
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=500)
        out = os.path.join(input_dir, f'out{i}.jpg')
        pix.save(out)
        print(f"  Page {i+1}: {pix.width}x{pix.height}")
    doc.close()
    return input_dir, output_dir, output_dir2


def run_reference(input_path, output_path, output_path2):
    """Run the reference algorithm, processing ALL pages (not just first)."""
    import numpy as np
    from segmenter import Segmenter
    import cv2
    from score_operator_new import Operater
    from conv_operation import conv_note, conv_note_expand
    from gramma_process import string_process
    import skimage.io as io

    img_paths = sorted(glob(f'{input_path}/*'))
    # Process ALL pages (reference code limits to [:1])
    print(f"Processing {len(img_paths)} images...")

    for i in range(len(img_paths)):
        print(f"\n{'='*60}")
        print(f"  Page {i+1}/{len(img_paths)}: {img_paths[i]}")
        print(f"{'='*60}")

        img = io.imread(img_paths[i])
        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = (img.shape[1], img.shape[0])

        result_image = Image.new("RGB", img_shape, "white")
        result_draw = ImageDraw.Draw(result_image)

        bin_img = 1 * (img > 127)
        segmenter = Segmenter(bin_img)

        imgs_with_staff = bin_img
        imgs_without_staff = segmenter.no_staff_img
        imgs_without_staff_raw = imgs_without_staff.copy()
        lines = segmenter.line_indices

        if len(lines) % 5 != 0:
            print(f"  SKIP: {len(lines)} lines detected, not divisible by 5")
            # Save original as output
            io.imsave(f'{output_path}/{i}.jpg', original_img)
            io.imsave(f'{output_path2}/{i}.jpg', original_img)
            continue

        lines_5line = np.array(lines).reshape(-1, 5)
        res_paper = np.zeros((lines_5line.shape[0], bin_img.shape[1], 19))

        all_clef = None

        for line in range(lines_5line.shape[0]):
            print(f"  Staff system {line+1}/{lines_5line.shape[0]}")

            line_5 = lines_5line[line, :]
            height = line_5[4] - line_5[0]
            start_line = line_5[0] - height // 2
            end_line = line_5[4] + height // 2

            img_line = imgs_with_staff[start_line:end_line, :]
            height = end_line - start_line

            cnt = 0
            all_conv = Operater(height)
            peaks_treble, res_paper, result_draw = conv_note(
                'treble', res_paper, all_conv, line, cnt, img_line,
                result_draw, start_line, end_line)
            peaks_bass, res_paper, result_draw = conv_note(
                'bass', res_paper, all_conv, line, cnt, img_line,
                result_draw, start_line, end_line)

            both_clef = np.append(peaks_treble, peaks_bass)
            if both_clef.shape[0] != 1:
                print(f"    Warning: {both_clef.shape[0]} clefs detected")

            if all_clef is None:
                all_clef = both_clef
            else:
                all_clef = np.append(all_clef, both_clef)

            # Bar detection
            height = (line_5[4] - line_5[0]) - 2
            start_line = line_5[0] + 1
            end_line = line_5[4] - 1
            img_line = imgs_with_staff[start_line:end_line, :]
            all_conv = Operater(round(height))
            if all_conv.expand == 1:
                peaks, res_paper, result_draw = conv_note_expand(
                    'bar', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)

            height = (line_5[4] - line_5[0]) / 4.0

            line_11 = [line_5[0] - 3*height, line_5[0] - 2*height,
                       line_5[0] - height,
                       line_5[0], line_5[1], line_5[2], line_5[3], line_5[4],
                       line_5[4] + height, line_5[4] + 2*height,
                       line_5[4] + 3*height]
            line_11 = np.array(line_11)
            line_21 = np.insert(line_11, np.arange(1, len(line_11)),
                                (line_11[:-1] + line_11[1:]) / 2)
            for k in range(21):
                line_21[k] = round(line_21[k])
            line_21 = line_21.astype(int)

            for cnt in range(19):
                start_line = line_21[cnt]
                end_line = line_21[cnt + 2]
                if start_line < 0 or end_line > np.shape(img)[0]:
                    continue
                height = end_line - start_line
                all_conv = Operater(round(height))
                img_line = imgs_without_staff[start_line:end_line, :]

                peaks, res_paper, result_draw = conv_note(
                    'natural', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)
                peaks, res_paper, result_draw = conv_note(
                    'flat', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)
                peaks, res_paper, result_draw = conv_note(
                    'sharp', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)
                peaks, res_paper, result_draw = conv_note(
                    'half', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)
                peaks, res_paper, result_draw = conv_note(
                    'whole', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)
                peaks, res_paper, result_draw = conv_note(
                    'quarte', res_paper, all_conv, line, cnt, img_line,
                    result_draw, start_line, end_line)

        white_img = np.ones_like(original_img, dtype=np.uint8) * 255
        res_paper_str, res_paper_str2 = string_process(
            imgs_with_staff, imgs_without_staff_raw,
            original_img, white_img, res_paper, lines_5line)

        io.imsave(f'{output_path}/{i}.jpg', res_paper_str)
        io.imsave(f'{output_path2}/{i}.jpg', res_paper_str2)
        print(f"  Page {i+1} done.")


def combine_pdfs(output_dir, pdf_name):
    """Combine output images into a single PDF."""
    image_files = sorted(glob(f'{output_dir}/*'))
    if not image_files:
        print(f"  No images in {output_dir}")
        return
    imgs = []
    for f in image_files:
        im = Image.open(f).convert('RGB')
        imgs.append(im)
    out_path = os.path.join(os.path.dirname(__file__), pdf_name)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:])
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    from PIL import ImageDraw
    import numpy as np

    # Change to repo dir so relative paths for templates work
    orig_cwd = os.getcwd()
    os.chdir(REPO_DIR)

    input_dir, output_dir, output_dir2 = prepare_input()
    run_reference(input_dir, output_dir, output_dir2)

    print("\nCombining output PDFs...")
    combine_pdfs(output_dir, 'erhu_reference_output.pdf')
    combine_pdfs(output_dir2, 'erhu_reference_output_2.pdf')

    os.chdir(orig_cwd)
    print("\nDone!")
