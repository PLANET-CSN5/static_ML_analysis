#value from paper 2259523

fp = open("results_2020_2_24_to_2020_3_13.csv")

firstfeat = []
firstfeatset = set()
secondfeat = []
secondfeatset = set()
for l in fp:
    sl = l.split(",")
    if len(sl) == 6:
        #print(sl[-2], sl[-1])
        firstfeat.append(sl[-2].split()[0])
        firstfeatset.add(sl[-2].split()[0])
        secondfeat.append(sl[-2].split()[1])
        secondfeatset.add(sl[-2].split()[1])

fp.close()

for v in firstfeatset:
    print(v, 100.0 * (firstfeat.count(v)/len(firstfeat)))