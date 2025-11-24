import matplotlib.pyplot as plt

def plot_geospatial(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df['longitude'], df['latitude'], s=1, alpha=0.5)
    plt.title("Geospatial Scatter — Data Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
