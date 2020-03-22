import sys, random

import pygame
from pygame.locals import *
import pymunk
from pymunk.pygame_util import DrawOptions
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import theano
import theano.tensor as tt
import numpy as np

def add_ball_static(space):
    mass = 1
    radius = 14
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = 150, 550
    shape = pymunk.Circle(body, radius)
    space.add(body, shape)
    return shape

def add_static_L(space,x,y,θ):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    body.position = (x, y)
    l1 = pymunk.Segment(body, (0, 0), (100, 0+θ), 2)
    space.add(l1)

def add_bucket(space):
    floor = pymunk.Body(body_type = pymunk.Body.STATIC)
    floor.position = (400, 200)

    left_wall = pymunk.Body(body_type = pymunk.Body.STATIC)
    left_wall.position = (400, 200)
    
    right_wall = pymunk.Body(body_type = pymunk.Body.STATIC)
    right_wall.position = (500, 200)
    
    l1 = pymunk.Segment(floor, (0, 0), (100, 0), 2)
    l2 = pymunk.Segment(left_wall, (0, 0), (0, 50), 2)
    l3 = pymunk.Segment(right_wall, (0, 50), (0, 0), 2)

    space.add(l1, l2, l3)
    return l1,l2,l3


@theano.compile.ops.as_op(itypes=[tt.lscalar, tt.lscalar, tt.lscalar,
                                  tt.lscalar, tt.lscalar, tt.lscalar],
                          otypes=[tt.lscalar])
def simulation(xl1, yl1, θl1,
               xl2, yl2, θl2):

    space = pymunk.Space()
    space.gravity = (0.0, -900.0)
    
    add_static_L(space, xl1, yl1, θl1)
    add_static_L(space, xl2, yl2, θl2)
    add_bucket(space)
    ball = add_ball_static(space)
    
    [space.step(1/50.0) for i in range(0,5000)]
    return np.array(ball.body.position.x, dtype=np.int64)

def visualize_simulation(xl1, yl1, θl1,
                         xl2, yl2, θl2):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Ball in the bucket game")    
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    lines = add_static_L(space, xl1, yl1, θl1)
    lines = add_static_L(space, xl2, yl2, θl2)
    lines = add_bucket(space)
    ball = add_ball_static(space)

    draw_options = DrawOptions(screen)

    stop = False
    while not stop:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        if ball.body.position.x > 410 and ball.body.position.y < 240:
            print("Ball in!")
            stop=True

        screen.fill((255,255,255))

        space.debug_draw(draw_options)
        
        space.step(1/50.0)

        pygame.display.flip()
        clock.tick(50)


def main():
    with pm.Model() as model:
        xl1 = pm.DiscreteUniform('xl1', lower=120,upper=121)
        yl1 = pm.DiscreteUniform('yl1', lower=350,upper=351)
        θl1 = pm.DiscreteUniform('θl1', lower=-20,upper=-19)


        xl2 = pm.DiscreteUniform('xl2', lower=0,upper=500)
        yl2 = pm.DiscreteUniform('yl2', lower=280,upper=281)
        θl2 = pm.DiscreteUniform('θl2', lower=0,upper=1)
        
        obs = pm.Normal('obs',
                        mu=simulation(xl1, yl1, θl1, xl2, yl2, θl2),
                        sigma=.001, observed=484)

        trace = pm.sample(10)

        
        def print_and_visualize(t):
            print(t)
            visualize_simulation(t['xl1'], t['yl1'], t['θl1'],
                                 t['xl2'], t['yl2'], t['θl2'])
        
        [print_and_visualize(t) for t in trace]
        
if __name__ == '__main__':
    sys.exit(main())
