import matplotlib.pyplot as plt
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
views = [10, 25, 40, 35, 60]
plt.plot(days, views, marker='o', color='blue', linestyle='--')
plt.title('My ML Journey Progress')
plt.xlabel('Learning Days')
plt.ylabel('Knowledge Gained / Views')
plt.show()

# imported matplotlib.pyplot as plt
# data is a days
# Its can be representing into x,y label 
# Its will show into graph structure