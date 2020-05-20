import sys

f = sys.argv[1]

with open(f) as handle:
    for new_line in handle:
        try:
            cmi = float(new_line.split()[0])
            if len(new_line.split()[1:]) < 20:
                continue
            if cmi >= 0.4:
                print(new_line.strip())
        except:
            pass
