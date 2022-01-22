print( 10 / 3)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

p = window(x, size=4)

print(list(p))