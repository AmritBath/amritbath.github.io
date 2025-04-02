# Rocket Throw Simulation

A single-page website that simulates a marble-throwing rocket’s journey home from the Moon, built with [PyScript](https://pyscript.net/). 

## Overview

- **Live Simulation**: Runs Python entirely in your browser.  
- **Parameters**: Choose how many marbles you want the astronaut to throw.  
- **Output**: Displays how long it takes to reach Earth, plus a fun summary of the journey.  
- **Animation**: Renders an interactive rocket animation illustrating the progress.

## Getting Started

1. **Clone or Download** this repository.
2. Place the `index.html` file at the root if you’re hosting on GitHub Pages (or open it locally in a browser).
3. Optionally add `space.jpg` and `rocket.png` into the same folder:
   - `space.jpg` = background starfield  
   - `rocket.png` = rocket image (rotated during the animation)

## Usage

1. Open `index.html` in your web browser.  
2. Input the **number of marbles** in the text box.  
3. Click **Run Simulation**.  
4. Watch the console messages for mission results, and see the animation appear below them.

## Folder Structure

```plaintext
|-- index.html
|-- space.jpg        (optional)
|-- rocket.png       (optional)
|-- README.md
