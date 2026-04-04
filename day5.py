import matplotlib.pyplot as plt

# Namma create panna data (GitHub profile views)
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
views = [10, 25, 40, 35, 60]

# Graph varaivom (Line Chart)
plt.plot(days, views, marker='o', color='blue', linestyle='--')

# Graph-ku Title and Labels set panrom
plt.title('My ML Journey Progress')
plt.xlabel('Learning Days')
plt.ylabel('Knowledge Gained / Views')

# Graph-ah screen-la kaata
plt.show()