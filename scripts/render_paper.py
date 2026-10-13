from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

if __name__=='__main__':
    document=pymupdf.open('original_snapshot/new_sakt__1_.pdf')
    Path('paper/rendered').mkdir(parents=True,exist_ok=True)
    for start in range(0,len(document),9):
        sheet=Image.new('RGB',(1200,1650),'white');draw=ImageDraw.Draw(sheet)
        for index in range(start,min(start+9,len(document))):
            pix=document[index].get_pixmap(matrix=pymupdf.Matrix(.6,.6))
            img=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);img.thumbnail((390,520))
            x=(index-start)%3*400;y=(index-start)//3*550
            sheet.paste(img,(x,y+20));draw.text((x+5,y+3),f'PDF page {index+1}',fill='black')
        sheet.save(f'paper/rendered/pages_{start+1}_{min(start+9,len(document))}.png')
