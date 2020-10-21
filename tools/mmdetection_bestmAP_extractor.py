import matplotlib.pyplot as plt
epoch = []
map = []
with open("test0410_aug_map_log.txt",'r') as f:
    line = f.readlines()
    for l in line:
        epoch.append(l.split(" ")[0])
        map.append(float(l.split(" ")[1]))


print("best map of Epoch!!" ,epoch[map.index(max(map))],":",max(map))
# plt.plot(map)
plt.plot(range(13,13+len(map)),map)
plt.grid()
plt.show()