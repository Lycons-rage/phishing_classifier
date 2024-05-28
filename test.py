a = {
    'a':1,
    'b':2,
    'e':-9,
    'd':5,
    'c':9
}

max_score = max(sorted(a.values()))

max_key = list(a.keys())[list(a.values()).index(max_score)]

print(max_key)