import random

def rand_add():
    a = random.randint(1,100)
    b = random.randint(1,100)
    answer = int(input(f'{a} + {b} = '))
    if answer == a+b:
        print("정답")
    else:
        print("오답")


rand_add()