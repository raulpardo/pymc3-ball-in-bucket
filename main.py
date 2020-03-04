import sys, random
## Example inspire by Oregon summer school lectures on probabilistic programming
## Code for simulation and visualization inspired from getting started tutorial in pymunk website

import pygame
from pygame.locals import *
import pymunk
from pymunk.pygame_util import DrawOptions
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

def add_ball(space):
    mass = 1
    radius = 14
    moment = pymunk.moment_for_circle(mass, 0, radius) # 1
    body = pymunk.Body(mass, moment) # 2
    x = random.randint(120, 480)
    body.position = x, 550 # 3
    shape = pymunk.Circle(body, radius) # 4
    space.add(body, shape) # 5
    return shape

def add_ball_static(space):
    mass = 1
    radius = 14
    moment = pymunk.moment_for_circle(mass, 0, radius) # 1
    body = pymunk.Body(mass, moment) # 2
    body.position = 150, 550 # 3
    shape = pymunk.Circle(body, radius) # 4
    space.add(body, shape) # 5
    return shape
def add_static_L(space,x,y,θ):
    body = pymunk.Body(body_type = pymunk.Body.STATIC)
    try:
        body.position = (x, y)
        l1 = pymunk.Segment(body, (0, 0), (100, 0+θ), 2)
        space.add(l1)
    except TypeError:
        pass
    # return l1

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

    space.add(l1, l2, l3) # 3
    return l1,l2,l3

def add_L(space):
    rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC) # 1
    rotation_center_body.position = (300, 300)

    rotation_limit_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    rotation_limit_body.position = (200,300)

    body = pymunk.Body(10, 10000) # 2
    body.position = (300, 300)
    l1 = pymunk.Segment(body, (-150, 0), (255.0, 0.0), 5.0)
    l2 = pymunk.Segment(body, (-150.0, 0), (-150.0, 50.0), 5.0)

    rotation_center_joint = pymunk.PinJoint(body,
                                            rotation_center_body,
                                            (0,0), (0,0)) # 3
    joint_limit = 25
    rotation_limit_joint = pymunk.SlideJoint(body,
                                             rotation_limit_body,
                                             (-100,0),
                                             (0,0),
                                             0, joint_limit) # 2
    space.add(l1, l2, body,
              rotation_center_joint, rotation_limit_joint)
    return l1,l2


# simulation(100,350,-20,250,280,0)  ## Good solution
def simulation(xl1, yl1, θl1,
               xl2, yl2, θl2):
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    add_static_L(space, xl1, yl1, θl1)
    add_static_L(space, xl2, yl2, θl2)
    add_bucket(space)
    ball = add_ball_static(space)    

    [space.step(1/50.0) for i in range(0,5000)]
    # print("( " + str(ball.body.position.x) + ", " + str(ball.body.position.y) + ")")
    return 1 if ball.body.position.x > 410 and ball.body.position.y < 240 else 20000000

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

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        if ball.body.position.x > 410 and ball.body.position.y < 240:
            print("Ball in!")

        screen.fill((255,255,255))

        space.debug_draw(draw_options)
        
        space.step(1/50.0)

        pygame.display.flip()
        clock.tick(50)


def main():
    # print(simulation(120,350,-20,250,280,0))
    # visualize_simulation(100,350,-20,250,280,0)

    with pm.Model() as model:
        xl1 = pm.Uniform('xl1', lower=0,upper=500)
        yl1 = pm.Uniform('yl1', lower=0,upper=500)
        θl1 = pm.Uniform('θl1', lower=-90,upper=90)


        xl2 = pm.Uniform('xl2', lower=0,upper=500)
        yl2 = pm.Uniform('yl2', lower=0,upper=500)
        θl2 = pm.Uniform('θl2', lower=-90,upper=90)

        obs = pm.Normal('obs', mu=simulation(xl1, yl1, θl1, xl2, yl2, θl2), sigma=0.001, observed=1)

        trace = pm.sample(10000)

        pm.traceplot(trace)
        plt.show()

        print(az.summary(trace))
        

if __name__ == '__main__':
    sys.exit(main())


