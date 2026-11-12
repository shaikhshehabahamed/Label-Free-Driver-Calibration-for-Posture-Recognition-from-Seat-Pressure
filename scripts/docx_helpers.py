"""python-docx utilities for the proposal (Letter, 1 in margins, Times New Roman 11 pt)."""
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"


def new_document():
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
    st = d.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.08
    for name, size, before, after in (("Heading 1", 14, 14, 6), ("Heading 2", 12, 10, 4), ("Heading 3", 11, 8, 3)):
        h = d.styles[name]
        h.font.name = FONT
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.italic = name == "Heading 3"
        h.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        h.paragraph_format.space_before = Pt(before)
        h.paragraph_format.space_after = Pt(after)
        h.paragraph_format.keep_with_next = True
    cap = d.styles["Caption"]
    cap.font.name = FONT
    cap.font.size = Pt(9.5)
    cap.font.italic = True
    cap.font.bold = False
    cap.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    cap.paragraph_format.space_after = Pt(8)
    add_page_number(d)
    return d


def add_page_number(d):
    p = d.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = text
        run._r.append(e)
    run.font.size = Pt(9)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_cell_margins(cell, top=40, bottom=40, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    m = OxmlElement("w:tcMar")
    for k, v in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        m.append(e)
    tcPr.append(m)


def para(d, text="", bold=False, italic=False, size=None, align=None, style=None, after=None, keep=False):
    p = d.add_paragraph(style=style)
    if text:
        add_runs(p, text, bold, italic, size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def add_runs(p, text, bold=False, italic=False, size=None):
    """text may contain **bold** and _italic_ markers (simple, non-nested)."""
    import re
    for part in re.split(r"(\*\*[^*]+\*\*|__[^_]+__)", text):
        if not part:
            continue
        b, i = bold, italic
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("__") and part.endswith("__"):
            part, i = part[2:-2], True
        r = p.add_run(part)
        r.bold, r.italic = b, i
        if size:
            r.font.size = Pt(size)
    return p


def bullets(d, items, size=None):
    for t in items:
        p = d.add_paragraph(style="List Bullet")
        add_runs(p, t, size=size)
        p.paragraph_format.space_after = Pt(2)


def table(d, header, rows, widths, font=8.5, header_fill="D9E2F3", zebra=False, align_right_from=None, bold_first_col=False):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.width = Inches(widths[j])
        shade(c, header_fill)
        set_cell_margins(c)
        c.paragraphs[0].paragraph_format.space_after = Pt(0)
        add_runs(c.paragraphs[0], h, bold=True, size=font)
    for i, r in enumerate(rows):
        cells = t.add_row().cells
        for j, v in enumerate(r):
            cells[j].width = Inches(widths[j])
            set_cell_margins(cells[j])
            if zebra and i % 2 == 1:
                shade(cells[j], "F5F5F5")
            p = cells[j].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            add_runs(p, str(v), bold=(bold_first_col and j == 0), size=font)
            if align_right_from is not None and j >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    # repeat header row on page breaks, keep the header with the first body rows, keep rows together
    trPr = t.rows[0]._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    e.set(qn("w:val"), "true")
    trPr.append(e)
    for r_ in t.rows[:1]:
        for c_ in r_.cells:
            for p_ in c_.paragraphs:
                p_.paragraph_format.keep_with_next = True
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
    return t


def caption(d, text):
    """figure captions stay below; a caption starting with 'Table' that directly follows a table is moved ABOVE it and kept with it."""
    p = d.add_paragraph(style="Caption")
    add_runs(p, text, italic=True, size=9.5)
    prev = p._p.getprevious()
    if text.startswith("Table") and prev is not None and prev.tag == qn("w:tbl"):
        prev.addprevious(p._p)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(3)
    return p


def figure(d, path, width_in, cap):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width_in))
    caption(d, cap)


def box(d, title, lines, fill="EEF3FA", width=6.5):
    t = d.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.width = Inches(width)
    shade(c, fill)
    set_cell_margins(c, 90, 90, 140, 140)
    p = c.paragraphs[0]
    add_runs(p, title, bold=True, size=11)
    p.paragraph_format.space_after = Pt(4)
    for ln in lines:
        q = c.add_paragraph()
        q.paragraph_format.space_after = Pt(3)
        q.paragraph_format.line_spacing = 1.05
        add_runs(q, ln, size=10)
    trPr = t.rows[0]._tr.get_or_add_trPr()          # a box never splits across pages
    cs = OxmlElement("w:cantSplit")
    cs.set(qn("w:val"), "true")
    trPr.append(cs)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def page_break(d):
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
