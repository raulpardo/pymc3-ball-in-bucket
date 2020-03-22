# Ball in the bucket game with probabilistic programming!

This is a simple python script that illustrates how to use
probabilistic programming (particularly PyMC3) to place bars in a 2
dimensional space so that a ball falling from the top ends up in a
bucket.

The program defines six random variables for the xy coordinates plus
inclination of each bar. We set uniform priors for these
variables---intuitively, this means that the bars could be located
anywhere in the space and with any inclination. Then we use a normal
distribution to constrain the final position of the ball. We set the
mean of the distribution as the x position where the ball should end
up and very low standard deviation---this forces the inference engine
to only accept samples in which the ball ends up in the bucket.

## Dependencies

In order to run this program you need to install
[PyMC3](https://docs.pymc.io/),
[Pymunk](http://www.pymunk.org/en/latest/), and
[Pygame](https://www.pygame.org/news).

## Run it!

To run it simply execute `python3 main.py` (after installing the
libraries above). The program will generate 10 possible solutions (xy
coordinates plus inclination) and visualize them. Additionally, it
will print in the standard output the solutions.

### Example

[Here](example-visualization.mp4) is an example visualization of one of the outputs that were
automatically computed by the script.


## Acknowledgment

I first saw this example in [one of the
lectures](https://www.youtube.com/watch?v=rZJJAobaQxM) on
Probabilistic Programming in the Oregon Programming Languages Summer
School by Sam Staton. The experiment in this script is basically a
copy of the example in the lecture but using
[PyMC3](https://docs.pymc.io/) (as probabilistic programming),
[Pymunk](http://www.pymunk.org/en/latest/) (as 2D physics engine) and
[Pygame](https://www.pygame.org/news) (for visualizing the
simulation).

The code for adding bars and balls as well as the visualization is
based on [this
example](http://www.pymunk.org/en/latest/tutorials/SlideAndPinJoint.html)
in the pymunk tutorials.
