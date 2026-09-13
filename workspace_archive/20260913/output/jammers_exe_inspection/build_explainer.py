"""Build the Chinese explanation PDF and reproducible illustrative figures."""
from pathlib import Path
import html
import math
import re
import subprocess
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, Preformatted)
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing, Circle, Line, String, PolyLine, Polygon
from reportlab.graphics import renderPDF
from reconstructed_rules import error_degrees

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'output/pdf/jammers-simulator_内部实现与隐藏规则详解.pdf'
TMP = ROOT/'tmp/pdfs/jammers_explainer'
FIG = HERE/'figures'
for directory in [OUT.parent, TMP, FIG]:
    directory.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('CN', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
BLUE = colors.HexColor('#165675')
TEAL = colors.HexColor('#16756c')
GREY = colors.HexColor('#677582')
LIGHT = colors.HexColor('#e5edf2')
ORANGE = colors.HexColor('#ba6339')

def label(d, x, y, text, size=9, color=GREY, anchor='start'):
    d.add(String(x,y,text,fontName='CN',fontSize=size,fillColor=color,textAnchor=anchor))

def figure(d, name):
    target=TMP/(name+'.pdf')
    renderPDF.drawToFile(d, str(target))
    subprocess.run(['/opt/homebrew/bin/pdftoppm','-r','160','-png','-singlefile',
                    str(target),str(FIG/name)],check=True,capture_output=True)

# Geometric support and the analytic radial CDF, not sampled official locations.
d=Drawing(500,205)
d.add(Circle(105,103,80,fillColor=colors.HexColor('#fae6d5'),strokeColor=ORANGE,strokeWidth=1))
d.add(Circle(105,103,80*1770/1800,fillColor=colors.HexColor('#e3f1ed'),strokeColor=TEAL,strokeWidth=1))
d.add(Line(105,103,184,103,strokeColor=TEAL,strokeWidth=1))
d.add(Circle(105,103,2,fillColor=TEAL,strokeColor=None))
label(d,130,109,'1770 m',10,TEAL)
label(d,105,70,'源的位置可行域',11,TEAL,'middle')
label(d,20,191,'1800 米目标圆与 30 米外圈',11,BLUE)
d.add(Line(178,133,215,153,strokeColor=ORANGE))
label(d,194,158,'30 m',10,ORANGE)
label(d,19,6,'圆的比例真实；外圈很窄。',9)
x0,y0,w,h=286,39,194,133
for t in [0,.25,.5,.75,1]:
    d.add(Line(x0,y0+t*h,x0+w,y0+t*h,strokeColor=LIGHT,strokeWidth=.5))
    label(d,x0-8,y0+t*h-3,f'{t:.2g}',8,anchor='end')
for t in [0,.5,1]:
    label(d,x0+t*w,y0-16,f'{t:.1f}',8,anchor='middle')
d.add(Line(x0,y0,x0+w,y0,strokeColor=GREY))
d.add(Line(x0,y0,x0,y0+h,strokeColor=GREY))
points=[]
for i in range(101):
    t=i/100
    points.extend([x0+t*w,y0+t*t*h])
d.add(PolyLine(points,strokeColor=BLUE,strokeWidth=2))
d.add(Circle(x0+.5*w,y0+.25*h,3,fillColor=ORANGE,strokeColor=None))
label(d,x0+8,y0+.25*h+12,'一半半径内约 25%',9,ORANGE)
label(d,286,191,'演练的径向累积分布',11,BLUE)
label(d,385,2,'半径 / 1770',9,anchor='middle')
figure(d,'domain')

d=Drawing(500,173)
x0,y0,w,h=42,35,440,105
for y in [-1,-.5,0,.5,1]:
    yp=y0+(y+1)/2*h
    d.add(Line(x0,yp,x0+w,yp,strokeColor=LIGHT,strokeWidth=.5))
    label(d,x0-8,yp-3,f'{y:g}',8,anchor='end')
for x in [0,150,300,450,600]:
    xp=x0+x/600*w
    d.add(Line(xp,y0,xp,y0+h,strokeColor=LIGHT,strokeWidth=.7))
    label(d,xp,y0-16,str(x),8,anchor='middle')
points=[]
for x in range(601):
    err=error_degrees(0x20260912,7,x,73)
    points.extend([x0+x/600*w,y0+(err+1)/2*h])
d.add(PolyLine(points,strokeColor=BLUE,strokeWidth=1.8))
label(d,6,154,'误差 / 度',9,BLUE)
label(d,220,154,'示意：固定种子、频道 7、y = 73 m',10,BLUE)
label(d,490,2,'观测位置 x / m',9,anchor='end')
figure(d,'noise')

d=Drawing(500,150)
cx,cy,r=240,76,58
# The clear disk is independent of the facing direction.
d.add(Circle(cx,cy,r,fillColor=colors.HexColor('#f4f6f8'),strokeColor=GREY,strokeWidth=.8))
pts=[cx,cy]
for i in range(41):
    a=-math.pi/2+math.pi*i/40
    pts.extend([cx+r*math.cos(a),cy+r*math.sin(a)])
pts.extend([cx,cy])
d.add(Polygon(pts,fillColor=colors.HexColor('#d5ece5'),strokeColor=None))
d.add(Line(cx,cy,cx+75,cy,strokeColor=TEAL,strokeWidth=1.5))
d.add(Line(cx+75,cy,cx+68,cy+4,strokeColor=TEAL,strokeWidth=1.5))
d.add(Line(cx+75,cy,cx+68,cy-4,strokeColor=TEAL,strokeWidth=1.5))
d.add(Circle(cx,cy,3,fillColor=TEAL,strokeColor=None))
d.add(Circle(cx-r/2,cy,3,fillColor=ORANGE,strokeColor=None))
label(d,8,121,'背面：检测不到也可能清除成功',11,ORANGE)
label(d,8,91,'例：观测点 (−10, 0)',10,ORANGE)
d.add(Line(160,89,cx-r/2,cy,strokeColor=ORANGE))
label(d,322,95,'朝向 +x 的正面',10,TEAL)
label(d,322,75,'绿色为正面覆盖方向',9,TEAL)
label(d,322,55,'灰圆：20 米清除范围',9)
label(d,cx,3,'源在 (0,0)；清除圆的背面同样有效。',9,anchor='middle')
figure(d,'directional')

styles={
 'body':ParagraphStyle('body',fontName='CN',fontSize=10.1,leading=16.3,spaceAfter=7,
                       textColor=colors.HexColor('#233847'),wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='CN',fontSize=24,leading=33,spaceAfter=15,
                        textColor=BLUE,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='CN',fontSize=17,leading=24,spaceAfter=15,
                     textColor=BLUE,wordWrap='CJK',keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=9.1,leading=14,wordWrap='CJK',
                       textColor=colors.HexColor('#233847')),
 'headcell':ParagraphStyle('headcell',fontName='CN',fontSize=9.1,leading=14,wordWrap='CJK',
                       textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8,leading=11.8,spaceAfter=9,
                       backColor=colors.HexColor('#f0f4f7'),borderPadding=9),
 'caption':ParagraphStyle('caption',fontName='CN',fontSize=8.5,leading=13,spaceAfter=8,
                          textColor=GREY,wordWrap='CJK'),
}

def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<font color="#165675">\1</font>',s)
    s=re.sub(r'`([^`]+)`',r'<font color="#165675">\1</font>',s)
    return s

def para(s, style='body'):
    return Paragraph(inline(s), styles[style])

text=(HERE/'内部实现与隐藏规则详解.md').read_text()
lines=text.splitlines()
story=[]
i=0
while i<len(lines):
    s=lines[i].strip()
    if not s:
        i+=1
        continue
    if s=='<!-- pagebreak -->':
        story.append(PageBreak());i+=1;continue
    if s.startswith('# '):
        title=s[2:].replace('jammers-simulator.exe ', 'jammers-simulator.exe<br/>')
        story.append(Paragraph(title,styles['title']));i+=1;continue
    if s.startswith('## '):
        story.append(para(s[3:],'h2'));i+=1;continue
    if s.startswith('```'):
        i+=1; block=[]
        while i<len(lines) and not lines[i].startswith('```'):
            block.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(block),styles['code']))
        i+=1;continue
    if s.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells):
                rows.append(cells)
            i+=1
        n=len(rows[0]); width=A4[0]-96
        widths=([width*.37,width*.63] if n==2 else [width*.28,width*.35,width*.37])
        contents=[[para(cell,'headcell' if row==0 else 'cell') for cell in values]
                  for row,values in enumerate(rows)]
        table=Table(contents,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
          ('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f7'),colors.white]),
          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
          ('LINEBELOW',(0,0),(-1,0),.5,BLUE),
        ]))
        story.extend([table,Spacer(1,11)]);continue
    match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',s)
    if match:
        from PIL import Image as PILImage
        path=HERE/match[2]
        with PILImage.open(path) as im: w,h=im.size
        story.append(Image(str(path),width=490,height=490*h/w))
        story.append(para(match[1],'caption'));i+=1;continue
    if s.startswith('- '):
        story.append(para('• '+s[2:]));i+=1;continue
    block=[s];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','<!--','![','- ')):
        block.append(lines[i].strip());i+=1
    story.append(para(' '.join(block)))

def decorate(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(.6)
    canvas.line(48,h-35,w-48,h-35)
    canvas.setFont('CN',8)
    canvas.setFillColor(GREY)
    canvas.drawString(48,h-26,'JAMMERS / 程序内部分析')
    canvas.drawRightString(w-48,h-26,'2026-09-12 · 静态证据')
    canvas.line(48,36,w-48,36)
    canvas.drawString(48,24,'仅对应已记录摘要的程序；重写版未经官方运行对照')
    canvas.drawRightString(w-48,24,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,
                      topMargin=53,bottomMargin=50,
                      title='jammers-simulator.exe 内部实现与隐藏规则详解',
                      author='Codex',subject='基于本地exe字节、规则对象及反汇编的中文解释')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
from pypdf import PdfReader
r=PdfReader(OUT)
print({'pdf':str(OUT),'pages':len(r.pages),'bytes':OUT.stat().st_size})
for j,p in enumerate(r.pages):
    t=p.extract_text()
    print(f'{j+1}: {len(t)} characters; '+t[:70].replace('\n',' / '))
