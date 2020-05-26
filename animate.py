import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, subplots
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np

class SimReader:
    
    def close(self):
        self.file.close()
    
    def get_population(self):
        if self.file.closed:
            return False
        healthy = '0'
        carriers = '1'
        sick = '2'
        immune = '3'
        count = [0, 0, 0, 0]
        colors = {
            # healthy: 'g',
            # carriers: 'm',
            # sick: 'r',
            # immune: 'b'
            healthy: 0,
            carriers: 100,
            sick: 30,
            immune: 40
        }
        population = {
            'x': [],
            'y': [],
            'health': []
        }
        for i in range(0, self.N):
            line = self.file.readline().split(' ')
            if not line or len(line) != 3:
                self.close()
                return False
            health = line[2][0]
            if health == healthy:
                count[0] += 1
            if health == carriers:
                count[1] += 1
            if health == sick:
                count[2] += 1
            if health == immune:
                count[3] += 1
            x = float(line[0])
            y = float(line[1])
            #population += [x, y, colors[health]]
            population['x'].append(x)
            population['y'].append(y)
            population['health'].append(colors[health])
        return (population, count)
            

    def __init__(self, file_path):
        #buffer_size = 1048576
    
        self.file_path = file_path
        print("Opening file...")
        self.file = open(self.file_path, "r")#, buffer_size)      
        preambule = self.file.readline().split(' ')
        self.N = int(preambule[0])
        self.DIM = float(preambule[1])
        self.iterations = int(preambule[2])
        self.gathering_points_n = int(preambule[3])
        self.gathering_points = []
        for i in range (0, self.gathering_points_n):
            line = self.file.readline().split(' ')
            self.gathering_points.append((float(line[0]), float(line[1])))
        print(f"{self.N} {self.DIM}")
        
    def __del__(self):
        self.close()


simreader = SimReader("output.sim")
    
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8,15))
population = simreader.get_population()

#ax1 = fig.add_subplot(211)
ax1.set_xlim([0, simreader.DIM])
ax1.set_ylim([0, simreader.DIM])

#ax2 = fig.add_subplot(212)
ax2.set_xlim([1, simreader.iterations])
ax2.set_ylim([0, simreader.N])

x = population[0]['x']
y = population[0]['y']
c = population[0]['health']

bx = [1]
by0 = [ population[1][0] ]
by1 = [ population[1][1] ]
by2 = [ population[1][2] ]
by3 = [ population[1][3] ]

gpx = []
gpy = []
for point in simreader.gathering_points:
    gpx.append(point[0])
    gpy.append(point[1])

ax1.scatter(gpx, gpy, c="m", marker='D', s=200)
scat = ax1.scatter(x, y, c=c, cmap='Set3', marker='.', s=10)
healthy, = ax2.plot(bx, by0, c='g')
carriers, = ax2.plot(bx, by1, c='y')
sick, = ax2.plot(bx, by2, c='r')
immune, = ax2.plot(bx, by3, c='b')

def update_frame(i, fig, scat, healthy, carriers, sick, immune):
    population = simreader.get_population()
    print(f"frame {i+1}")
    if not population:
        return
    x = population[0]['x']
    y = population[0]['y']
    c = population[0]['health']
    
    bx.append(bx[-1] + 1)
    by0.append(population[1][0])
    by1.append(population[1][1])
    by2.append(population[1][2])
    by3.append(population[1][3])
    
    healthy.set_xdata(bx)
    healthy.set_ydata(by0)
    carriers.set_xdata(bx)
    carriers.set_ydata(by1)
    sick.set_xdata(bx)
    sick.set_ydata(by2)
    immune.set_xdata(bx)
    immune.set_ydata(by3)
    
    scat.set_offsets(np.c_[x, y])
    scat.set_array(np.array(c))

anim = animation.FuncAnimation(fig, update_frame, fargs=(fig, scat, healthy, carriers, sick, immune), frames=simreader.iterations-1, interval=250, repeat=False)
plt.show()
