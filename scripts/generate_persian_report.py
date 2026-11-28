# -*- coding: utf-8 -*-
"""
Builds a Persian-language PDF report (for presentation use) explaining the
physics, math, and code of this repository, with rendered formulas (as
matplotlib-mathtext images), the B Nazanin font for Persian prose, and the
repo's three result figures embedded.

Usage:
    python3 scripts/generate_persian_report.py
Output:
    report/Persian_Report.pdf
"""
from __future__ import annotations

import os
import sys

from fpdf import FPDF
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from report_content_fa import CONTENT  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(HERE, "assets", "fonts")
MPL_FONT_DIR = "/usr/local/lib/python3.11/dist-packages/matplotlib/mpl-data/fonts/ttf"
OUT_DIR = os.path.join(HERE, "report")
OUT_PATH = os.path.join(OUT_DIR, "Persian_Report.pdf")

PAGE_W, PAGE_H = 210, 297  # A4 mm
MARGIN = 20
CONTENT_W = PAGE_W - 2 * MARGIN

TITLE = "یادگیری طبقه‌بندی فازهای کوانتومی ماده با چند اندازه‌گیری"
SUBTITLE = "گزارش فنی برای ارائه: فیزیک، روش و پیاده‌سازی نرم‌افزاری"
PAPER_REF = (
    "بر اساس: M. Khosrojerdi, J. L. Pereira, A. Cuccoli, L. Banchi — "
    "Quantum Sci. Technol. 10, 025006 (2025)"
)


class ReportPDF(FPDF):
    def __init__(self, toc_pages=None):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.add_font("BNazanin", "", os.path.join(FONT_DIR, "BNazanin.ttf"))
        self.add_font("BTitrBold", "", os.path.join(FONT_DIR, "BTitrBold.ttf"))
        self.add_font("BZar", "", os.path.join(FONT_DIR, "BZar.ttf"))
        self.add_font("DejaVu", "", os.path.join(MPL_FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(MPL_FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVuMono", "", os.path.join(MPL_FONT_DIR, "DejaVuSansMono.ttf"))
        self.set_fallback_fonts(["DejaVu"], exact_match=False)
        self.set_text_shaping(True)
        self.set_font("BNazanin", size=12)
        self.toc_pages = toc_pages or {}  # title -> page number (filled pass 2)
        self.recorded_toc = []  # [(level, title, page_no)] filled every pass
        self.chapter_no = 0

    # ---- low level helpers -------------------------------------------------
    def footer(self):
        self.set_y(-15)
        self.set_font("BNazanin", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def _rtl_cell(self, h, text, font="BNazanin", size=12, style="", color=(0, 0, 0)):
        self.set_font(font, style, size)
        self.set_text_color(*color)
        self.set_x(MARGIN)
        self.multi_cell(CONTENT_W, h, text, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    # ---- block renderers ----------------------------------------------------
    def block_h1(self, title):
        self.add_page()
        self.recorded_toc.append((1, title, self.page_no()))
        self.ln(2)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.8)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(4)
        self._rtl_cell(11, title, font="BTitrBold", size=19, color=(20, 40, 90))
        self.ln(2)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.4)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(6)

    def block_h2(self, title):
        if self.get_y() > PAGE_H - MARGIN - 30:
            self.add_page()
        self.recorded_toc.append((2, title, self.page_no()))
        self.ln(3)
        self._rtl_cell(9, title, font="BTitrBold", size=14, color=(30, 60, 120))
        self.ln(1)

    def block_h3(self, title):
        if self.get_y() > PAGE_H - MARGIN - 25:
            self.add_page()
        self.ln(2)
        self._rtl_cell(8, title, font="BTitrBold", size=12.5, color=(50, 50, 50))

    def block_p(self, text):
        self._rtl_cell(7.4, text, font="BNazanin", size=11.5)
        self.ln(2.5)

    def block_p_ltr(self, text):
        """Left-to-right paragraph (e.g. an English citation), unshaped.

        A wrapped line of an Arabic-shaped RTL multi_cell that ends up being
        pure-Latin/numeric (no Arabic anchor characters) can trigger a
        reproducible fpdf2/HarfBuzz glyph-corruption bug. Rendering such
        text with shaping disabled and left-aligned avoids it entirely.
        """
        if self.get_y() > PAGE_H - MARGIN - 25:
            self.add_page()
        self.set_font("DejaVu", size=11)
        self.set_x(MARGIN)
        self.set_text_shaping(False)
        self.multi_cell(CONTENT_W, 7, text, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_shaping(True)
        self.ln(2.5)

    def block_bullets(self, items):
        for it in items:
            self._rtl_cell(7, "-   " + it, font="BNazanin", size=11.5)
        self.ln(2)

    def block_numbered(self, items):
        digits = "۰۱۲۳۴۵۶۷۸۹"
        for i, it in enumerate(items, 1):
            num = "".join(digits[int(c)] for c in str(i))
            self._rtl_cell(7, f"{num}.   " + it, font="BNazanin", size=11.5)
        self.ln(2)

    def block_formula(self, path, caption):
        with Image.open(path) as im:
            w_px, h_px = im.size
        dpi = 340.0
        w_mm = w_px / dpi * 25.4
        h_mm = h_px / dpi * 25.4
        max_w = CONTENT_W * 0.86
        if w_mm > max_w:
            scale = max_w / w_mm
            w_mm *= scale
            h_mm *= scale
        if self.get_y() + h_mm + 10 > PAGE_H - MARGIN:
            self.add_page()
        self.ln(2)
        x = MARGIN + (CONTENT_W - w_mm) / 2
        self.image(path, x=x, y=self.get_y(), w=w_mm, h=h_mm)
        self.set_y(self.get_y() + h_mm + 2)
        if caption:
            self._rtl_cell(6, caption, font="BNazanin", size=10)
        self.ln(3)

    def block_figure(self, path, caption, width_mm):
        with Image.open(path) as im:
            w_px, h_px = im.size
        w_mm = width_mm
        h_mm = w_mm * h_px / w_px
        if self.get_y() + h_mm + 14 > PAGE_H - MARGIN:
            self.add_page()
        self.ln(2)
        x = MARGIN + (CONTENT_W - w_mm) / 2
        self.set_draw_color(180, 180, 180)
        self.rect(x - 1, self.get_y() - 1, w_mm + 2, h_mm + 2)
        self.image(path, x=x, y=self.get_y(), w=w_mm, h=h_mm)
        self.set_y(self.get_y() + h_mm + 3)
        if caption:
            self._rtl_cell(6.2, caption, font="BNazanin", size=10.5, style="")
        self.ln(3)

    def block_code(self, text):
        lines = text.split("\n")
        line_h = 5.0
        box_h = line_h * len(lines) + 6
        if self.get_y() + box_h > PAGE_H - MARGIN:
            self.add_page()
        y0 = self.get_y()
        self.set_fill_color(244, 244, 248)
        self.set_draw_color(210, 210, 220)
        self.rect(MARGIN, y0, CONTENT_W, box_h, style="DF")
        self.set_xy(MARGIN + 3, y0 + 3)
        self.set_font("DejaVuMono", size=8.6)
        for ln in lines:
            self.set_x(MARGIN + 3)
            self.cell(CONTENT_W - 6, line_h, ln, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_y(y0 + box_h + 4)

    def block_table(self, headers, rows):
        n = len(headers)
        col_w = CONTENT_W / n
        row_h = 8.5
        if self.get_y() + row_h * (len(rows) + 1) > PAGE_H - MARGIN and self.get_y() > MARGIN + 20:
            self.add_page()
        # header row (right-to-left: headers[0] rightmost)
        y = self.get_y()
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("BNazanin", size=10.5)
        for i, htext in enumerate(headers):
            x = MARGIN + CONTENT_W - col_w * (i + 1)
            self.set_xy(x, y)
            self.multi_cell(col_w, row_h, htext, align="C", border=1, fill=True,
                             new_x="LEFT", new_y="TOP", max_line_height=row_h)
        self.set_text_color(0, 0, 0)
        self.set_y(y + row_h)
        fill = False
        for row in rows:
            # measure needed height
            self.set_font("BNazanin", size=10)
            heights = []
            for i, cell_text in enumerate(row):
                lines = self.multi_cell(col_w - 2, 5.2, str(cell_text), align="C",
                                         dry_run=True, output="LINES")
                heights.append(max(1, len(lines)) * 5.2 + 2)
            rh = max(row_h, max(heights))
            if self.get_y() + rh > PAGE_H - MARGIN:
                self.add_page()
            y = self.get_y()
            self.set_fill_color(245, 247, 252) if fill else self.set_fill_color(255, 255, 255)
            for i, cell_text in enumerate(row):
                x = MARGIN + CONTENT_W - col_w * (i + 1)
                self.set_xy(x, y)
                self.multi_cell(col_w, rh, str(cell_text), align="C", border=1,
                                 fill=True, new_x="LEFT", new_y="TOP", max_line_height=5.2)
            self.set_y(y + rh)
            fill = not fill
        self.ln(4)

    def block_space(self, mm):
        self.ln(mm)

    def block_mono_line(self, text):
        """A single LTR line (URL/path), left-aligned, auto-shrunk to fit."""
        size = 11.0
        self.set_font("DejaVu", size=size)
        while self.get_string_width(text) > CONTENT_W - 4 and size > 6:
            size -= 0.5
            self.set_font("DejaVu", size=size)
        if self.get_y() + 8 > PAGE_H - MARGIN:
            self.add_page()
        self.set_x(MARGIN)
        self.set_text_shaping(False)
        self.cell(CONTENT_W, 6.5, text, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_shaping(True)
        self.ln(1)

    def block_hr(self):
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    # ---- page-level builders -------------------------------------------------
    def cover_page(self):
        self.add_page()
        # NOTE: a vector rect()/line() drawn as the very first content operation
        # of the whole document (before any text has been shaped) corrupts the
        # glyph rendering of subsequently-shaped text on later pages — this is a
        # reproducible fpdf2/HarfBuzz interaction bug, not a content issue. Using
        # a solid-color raster image for the title banner instead of pdf.rect()
        # avoids triggering the bug while looking identical.
        banner = Image.new("RGB", (10, 10), (20, 40, 90))
        self.image(banner, x=0, y=0, w=PAGE_W, h=70)
        self.set_y(26)
        self.set_text_color(255, 255, 255)
        self.set_font("BTitrBold", size=23)
        self.multi_cell(CONTENT_W, 12, TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("BNazanin", size=13)
        self.multi_cell(CONTENT_W, 8, SUBTITLE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

        self.set_y(100)
        self.set_font("BNazanin", size=12)
        self.multi_cell(CONTENT_W, 7.5, PAPER_REF, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("DejaVu", size=10.5)
        self.cell(0, 6, "doi:10.1088/2058-9565/ada79b   |   arXiv:2404.18231", align="C",
                  new_x="LMARGIN", new_y="NEXT")

        cover_img = os.path.join(HERE, "figures", "figure2.png")
        if os.path.exists(cover_img):
            with Image.open(cover_img) as im:
                w_px, h_px = im.size
            w_mm = 150
            h_mm = w_mm * h_px / w_px
            x = (PAGE_W - w_mm) / 2
            y_img = 138
            self.set_draw_color(190, 190, 190)
            self.rect(x - 1, y_img - 1, w_mm + 2, h_mm + 2)
            self.image(cover_img, x=x, y=y_img, w=w_mm, h=h_mm)
            self.set_y(y_img + h_mm + 4)
            self.set_font("BNazanin", size=9.5)
            self.set_text_color(130, 130, 130)
            self.cell(0, 5, "شکل ۲ مقاله: کران تعمیم‌پذیری و ساختار هسته", align="C",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)

        self.set_y(PAGE_H - 35)
        self.set_font("BNazanin", size=10)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, "بازتولید مستقل مبتنی بر متن مقاله — بدون استفاده از کد یا داده‌ی نویسندگان",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 6, "آبان ۱۴۰۴", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def toc_page(self):
        self.add_page()
        self._rtl_cell(12, "فهرست مطالب", font="BTitrBold", size=18, color=(20, 40, 90))
        self.ln(4)
        self.set_draw_color(30, 60, 120)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(6)
        for level, title, pageno in self.toc_pages.get("entries", []):
            if level == 1:
                self.set_font("BNazanin", size=12.5)
                y_before = self.get_y()
                num = self._to_fa_digits(str(pageno))
                self.set_x(MARGIN)
                self.cell(14, 7.5, num, align="L")
                self.set_x(MARGIN)
                self.multi_cell(CONTENT_W - 16, 7.5, title, align="R",
                                 new_x="LMARGIN", new_y="NEXT")
            else:
                self.set_font("BNazanin", size=10.5)
                num = self._to_fa_digits(str(pageno))
                self.set_x(MARGIN)
                self.cell(14, 6.5, num, align="L")
                self.set_x(MARGIN)
                self.set_text_color(70, 70, 70)
                self.multi_cell(CONTENT_W - 16, 6.5, "   " + title, align="R",
                                 new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)

    @staticmethod
    def _to_fa_digits(s):
        digits = "۰۱۲۳۴۵۶۷۸۹"
        return "".join(digits[int(c)] if c.isdigit() else c for c in s)

    def render_body(self):
        dispatch = {
            "h1": self.block_h1,
            "h2": self.block_h2,
            "h3": self.block_h3,
            "p": self.block_p,
            "p_ltr": self.block_p_ltr,
            "bullets": self.block_bullets,
            "numbered": self.block_numbered,
            "space": self.block_space,
            "hr": lambda: self.block_hr(),
            "mono_line": self.block_mono_line,
        }
        for block in CONTENT:
            kind = block[0]
            if kind == "formula":
                _, path, caption = block
                self.block_formula(os.path.join(HERE, path) if not os.path.isabs(path) else path, caption)
            elif kind == "figure":
                _, path, caption, width_mm = block
                self.block_figure(os.path.join(HERE, path) if not os.path.isabs(path) else path, caption, width_mm)
            elif kind == "code":
                self.block_code(block[1])
            elif kind == "table":
                self.block_table(block[1], block[2])
            elif kind in dispatch:
                fn = dispatch[kind]
                if kind in ("p", "code"):
                    fn(block[1])
                elif kind in ("bullets", "numbered"):
                    fn(block[1])
                elif kind == "space":
                    fn(block[1])
                elif kind == "hr":
                    fn()
                else:
                    fn(block[1])
            else:
                raise ValueError(f"unknown block kind {kind!r}")


def build(toc_pages=None):
    pdf = ReportPDF(toc_pages=toc_pages)
    pdf.cover_page()
    pdf.toc_page()
    pdf.render_body()
    return pdf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pass 1: render once to discover each heading's page number.
    pass1 = build(toc_pages=None)
    entries = pass1.recorded_toc

    # Pass 2: render again, this time filling the TOC page with real numbers.
    pass2 = build(toc_pages={"entries": entries})
    pass2.output(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({pass2.page_no()} pages)")


if __name__ == "__main__":
    main()
