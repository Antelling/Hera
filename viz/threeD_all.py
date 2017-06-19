def show():
    import data, math

    X, y = data.get.people_xy()

    from sklearn.cluster import AgglomerativeClustering as Clusterer
    clusterer = Clusterer(n_clusters=16)
    clusterer = clusterer.fit(X)
    print(clusterer.labels_)

    from sklearn.decomposition import PCA, ProjectedGradientNMF, TruncatedSVD
    from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE
    X_transformed = SpectralEmbedding(n_components=3).fit_transform(X)

    plt_x = []
    plt_y = []
    plt_z = []
    plt_c = list(map(lambda x:x+1, clusterer.labels_))
    for a in X_transformed:
        plt_x.append(a[0])
        plt_y.append(a[1])
        plt_z.append(a[2])


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def time_mod(x):
        return 3.85/(1 + 5 * math.exp(-.05 * x))

    couples = data.get.couples_raw()
    for couple in couples:
        x_index = y.index(couple["male"])
        y_index = y.index(couple["female"])
        a = X_transformed[x_index]
        b = X_transformed[y_index]
        line = ax.plot([a[0], b[0]], [a[1], b[1]], zs=[a[2], b[2]], c="black")
        plt.setp(line, linewidth=time_mod(couple["length"]))

    ax.scatter(plt_x, plt_y, plt_z, c=plt_c)

    plt.show()

show()