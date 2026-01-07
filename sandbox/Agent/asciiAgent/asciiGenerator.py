from langchain.tools import tool;
import pyfiglet;
import random;

@tool
def asciiArtFromText(text: str, fontStyle: str = "random") -> str:
     """
     Generate ASCII art from a text string using a specified font style.

     Args:
          text (str): 
               The input text to convert into ASCII art.
               Example: "Hello World"

          fontStyle (str, optional): 
               The pyfiglet font to use when generating ASCII art.
               If set to "random", the tool will randomly select a font
               from a predefined list of supported styles. The predefined styles include:
               - "slant": Slanted text style
               - "3-d": 3D block text style
               - "3x5": Custom 3x5 dimensions
               - "5lineoblique": Oblique text style
               - "alphabet": Made of other alphabets
               - "banner3-D": 3D banner style
               - "doh" : Doh style
               - "isometric1": Isometric text style
               - "letters": Made of letters
               - "alligator": Alligator style
               - "dotmatrix": Dot matrix style
               - "bubble": Bubble text style
               - "bulbhead": Bulb head style
               - "digital": Digital style
               Default is "random".

     Returns:
          str:
               A string containing the generated ASCII art representation
               of the input text.

     Tool Behavior:
          - If `fontStyle` is "random", a font is randomly chosen.
          - Uses the pyfiglet library to render ASCII art.
          - Returns only the ASCII art string (no additional metadata).

     Example Usages (LangChain Agent):
          asciiArtFromText("LangChain", fontStyle="slant");
          asciiArtFromText("Hello World", fontStyle="random");
          
     When to Use:
          -    Use this tool when you need to create visually appealing ASCII art
               representations of text for display in console applications,
               chatbots, or any text-based interface.
          -   Ideal for enhancing user interaction with creative text designs.
          -   Suitable for generating banners, headings, or decorative text elements.
          -   Can be used when the user directly requests ASCII art generation.
     """

     # Supported ASCII font styles for pyfiglet
     styles = [
          "slant",
          "3-d",
          "3x5",               # supports custom dimensions
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

     # Randomly select a font if none is explicitly provided
     if fontStyle == "random":
          fontStyle = random.choice(styles)

     # Generate ASCII art using pyfiglet
     asciiArt = pyfiglet.figlet_format(text, font=fontStyle)

     # Return ASCII art as a plain string for the agent to use
     return asciiArt
          

if __name__ == "__main__":
     # Example Usage
     textArt = asciiArtFromText("Oreeeee!", fontStyle="slant");
     print(textArt);