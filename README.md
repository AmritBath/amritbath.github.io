# amritbath.github.io

Rocket Throw Simulation

This project is a single-page website, built using PyScript, that simulates a rocket propelled by the throwing of marbles. The goal is to see how long it takes for the rocket to travel a specified distance (for example, from the Moon to Earth) by repeatedly throwing marbles.

How It Works

PyScript: A framework that embeds a Python environment in the browser, allowing you to run Python scripts directly in HTML without a separate server.
Marble-Throwing Mechanics: The rocket repeatedly throws marbles to generate thrust. It goes through "throwing phases" and "resting phases" until it reaches its target distance.
User Input: You can enter the number of marbles you start with. More marbles generally translate to more throws and (potentially) faster travel.
Simulation: The code uses a custom ODE solver (via the Runge-Kutta 4 method) to track position, velocity, and mass over time.
Animation: At the end of the simulation, a short Matplotlib animation is generated to give a rough visual representation of how far the rocket has travelled.
Getting Started

Download/Clone the index.html file containing the PyScript code.
Open the index.html file in a modern web browser (e.g. Chrome, Firefox, or Edge).
Input a number of marbles (default: 1000) in the provided box.
Click the Run Simulation button to start the marble-throwing simulation.
Prerequisites

An internet connection (to load the PyScript assets from the official CDN).
A modern browser that supports WebAssembly (since PyScript needs it).
Optional images (rocket.png and space.jpg) in the same folder as index.html to add more visual flair to the animation.
Folder Structure

.
├── index.html        # Main HTML page containing all code and PyScript
├── rocket.png        # (Optional) Rocket image used in animation
└── space.jpg         # (Optional) Background image for the animation
If rocket.png or space.jpg are not found, the simulation will run anyway, but a placeholder or blank background will be shown instead.
Explanation of Key Sections

Simulation Parameters
D: Target distance in metres (e.g. Earth-Moon distance).
dt: Time step for the numerical integration.
R0, beta, m, u, M_dry, R_threshold, tau: Various constants controlling throw rate, mass consumption, thrust velocity, rest thresholds, and rest duration.
simulate_full()
The main function that repeatedly calls the throwing_phase() and resting_phase() functions. It returns arrays of time, position, velocity, acceleration, and mass so you can track the rocket's progress.
animate_rocket(distance)
A simple function that uses matplotlib.animation.FuncAnimation to animate a rocket image moving across the screen, scaled to the final distance of the simulation (in kilometres).
generate_fun_summary()
Prints a light-hearted summary once the simulation completes. Includes random messages about fatigue and hunger levels, as well as an imaginary “best friend” marble.
Troubleshooting

No animation appears: Make sure your browser console has no errors. Some older browsers might not fully support PyScript.
Images not found: If rocket.png or space.jpg are missing, the code will gracefully fall back on defaults. You can ignore missing image errors if you don’t want those extra visuals.
Slow loading: The first time you open the page, PyScript must download Python packages over the internet. Subsequent loads are generally faster.
Contributing

Fork or clone this repository.
Make your improvements or fix bugs in a new branch.
Submit a pull request (PR) describing your changes.
Licence

This project is provided under the MIT Licence, so feel free to modify and distribute it as you wish. If you reuse it, a shout-out is always appreciated!
