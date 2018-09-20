import svbfile
from matplotlib import pyplot as plt

svbfile = svbfile.SVBFile(r'example.svb')

plt.figure()
plt.clf()
for name, channel in svbfile.items():
    plt.plot(channel.Time, channel.Values, label=name)

plt.legend()
plt.grid()
    