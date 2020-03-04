import sys, random
import pygame
from pygame.locals import *
import pymunk #1
from pymunk.pygame_util import DrawOptions

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
    body = pymunk.Body(body_type = pymunk.Body.STATIC) # 1
    body.position = (x, y)
    l1 = pymunk.Segment(body, (0, 0), (100, 0+θ), 2) # 2
    # l1 = pymunk.Segment(body, (-150, 0), (-150, 50), 5)

    space.add(l1)#, l2) # 3
    return l1#,l2

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


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Joints. Just wait and the L will tip over")
    clock = pygame.time.Clock()

    space = pymunk.Space() #2
    space.gravity = (0.0, -900.0)

    lines = add_static_L(space,100,350,-20)
    lines = add_static_L(space,250,280,0)
    lines = add_bucket(space)
    ball = add_ball_static(space)
    # balls = []
    draw_options = DrawOptions(screen)

    ticks_to_next_ball = 10
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        if ball.body.position.x > 410 and ball.body.position.y < 240:
            # print("( " + str(ball.body.position.x) + ", " + str(ball.body.position.y) + ")")
            print("Ball in!")
            sys.exit(0)
                    

        # ticks_to_next_ball -= 1
        # if ticks_to_next_ball <= 0:
        #     ticks_to_next_ball = 25
        #     ball_shape = add_ball_static(space)
        #     balls.append(ball_shape)            

        screen.fill((255,255,255))

        # balls_to_remove = []
        # for ball in balls:
        #     if ball.body.position.y < 150:
        #         balls_to_remove.append(ball)

        # for ball in balls_to_remove:
        #     space.remove(ball, ball.body)
        #     balls.remove(ball)

        space.debug_draw(draw_options)
        
        space.step(1/50.0) #3       

        pygame.display.flip()
        clock.tick(50)

if __name__ == '__main__':
    sys.exit(main())


