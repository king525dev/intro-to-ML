import os
import pyfiglet
import ascii_magic

def asciiArtFromText(text, fontStyle="random"):

          styles = [
               "slant",
               "3-d",
               "3x5 (you can set custom dimensions)",
               "5lineoblique", 
               "alphabet",
               "banner3-D",
               "doh", 
               "isometric1",
               "letters",
               "alligator",
               "dotmatrix",
               "bubble",
               "bulbhead",
               "digital"
          ]
          
          if fontStyle == "random":
               import random
               fontStyle = random.choice(styles)

          asciiArt = pyfiglet.figlet_format(text,font=fontStyle);

          return asciiArt;
          
def asciiArtFromImage(imagePath):  
     # Generate Ascii Art from Image
     output = ascii_magic.from_image_file(imagePath, columns=80, char="#");
     output.to_html();
     print("File Created")

def randomAsciiImage():
     # Generate random Ascii Image
     try:
          output = ascii_magic.from_url('https://picsum.photos/');
     except OSError as e:
          print(f'Could not load the image, server said: {e.code} {e.msg}');
          
     output.to_html();
     print("File Created")
     

if __name__ == "__main__":
     # Example Usage
     textArt = asciiArtFromText("Hello World!", fontStyle="slant");
     print(textArt);

     # Example Image to Ascii Art
     imagePath = "example.jpg"  # Replace with your image path
     if os.path.exists(imagePath):
          asciiArtFromImage(imagePath);
     else:
          print(f"Image file '{imagePath}' not found. Skipping image to ASCII art conversion.");

     # Example Random Ascii Image
     randomAsciiImage();