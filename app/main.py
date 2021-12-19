import pandas as pd
from matplotlib import pyplot as plt
from flask import (
    Flask,
    render_template
)
from sklearn.tree import DecisionTreeClassifier
import config
import joblib

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

stats = pd.read_csv("Stephen Curry Regularseason Stats.csv")
stats_view = pd.read_csv("Stephen Curry Regularseason Stats.csv")
stats.columns = [
    "season_year",
    "season_div",
    "date",
    "opp",
    "result",
    "t_score",
    "o_score",
    "min",
    "pts",
]
numeric_columns = stats.loc[:, ["result",
                                "t_score",
                                "o_score",
                                "min",
                                "pts",
                                ], ]

stats.result = stats['result'].replace(to_replace="L", value="0")
stats.result = stats['result'].replace(to_replace="W", value="1")

info = stats.describe(percentiles=[])

app = Flask(
    __name__, static_folder=config.STATIC_FOLDER, template_folder=config.TEMPLATE_FOLDER
)


@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template(
        "index.html",
        tables=[stats_view.to_html(classes="data")],
        titles=stats_view.columns.values,)


@app.route("/view", methods=("POST", "GET"))
def html_table1():
    return render_template("view.html",
                           tables=[info.to_html(classes="data")],)


@app.route("/min_pts_season_find", methods=("POST", "GET"))
def html_min_pts_season_find():
    season, pts = min_pts_season_find()
    return render_template("min_pts_season.html",
                           season=season,
                           pts=pts)


@app.route("/max_min_season_find", methods=("POST", "GET"))
def html_max_min_season_find():
    season, min = max_min_season_find()
    return render_template("max_min_season.html",
                           season=season,
                           min=int(min))


@app.route("/mean_pts", methods=("POST", "GET"))
def html_mean_pts():
    opp = mean_pts_find()
    return render_template("mean_pts.html",
                           opp=opp,)


@app.route("/classification", methods=("POST", "GET"))
def html_clustering():
    names, values, score = classification()
    print(names)
    print(list(values))
    clusters = pd.DataFrame(data=[values], columns=names, index=["Важность"])
    return render_template(
        "classification.html",
        score=round(score, 5),
        tables=[clusters.to_html(classes="data")])


def min_pts_season_find():
    """поиск сезона с минимальным средним количеством очков"""
    groups = stats.groupby('season_year', as_index=False)['season_year', 'pts'].mean()
    min_pts_season_year = groups[groups['pts'] == groups['pts'].min()]

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_title('Распределение очков по сезонам')
    plt.style.context('bmh')
    ax.set_xlabel('Сезон')
    ax.set_ylabel('Количество очков')

    seasons = groups['season_year'].values.flatten().tolist()
    pts = groups['pts'].values.flatten().tolist()
    plt.bar(seasons, pts)

    plt.grid('on')
    plt.savefig('static/min_pts_season.png')
    return min_pts_season_year['season_year'].iloc[0], min_pts_season_year['pts'].iloc[0]


def max_min_season_find():
    """поиск сезона с максимальным средним временем на паркете"""
    groups = stats.groupby('season_year', as_index=False)['season_year', 'min'].mean()
    max_min_season_year = groups[groups['min'] == groups['min'].max()]

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_title('Среднее время на паркете по сезонам')
    plt.style.context('bmh')
    ax.set_xlabel('Сезон')
    ax.set_ylabel('Среднее время')

    seasons = groups['season_year'].values.flatten().tolist()
    pts = groups['min'].values.flatten().tolist()
    plt.bar(seasons, pts)

    plt.grid('on')
    plt.savefig('static/max_min_season.png')
    return max_min_season_year['season_year'].iloc[0], max_min_season_year['min'].iloc[0]


def mean_pts_find():
    """поиск среднего счета против каждой команды"""
    groups = stats.groupby('opp', as_index=False)['opp', 'pts'].mean()
    max_min_season_year = groups[groups['pts'] == groups['pts'].max()]

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    ax.set_title('Среднее количество очков против каждой команды')
    plt.style.context('bmh')
    ax.set_xlabel('Команда')
    ax.set_ylabel('Средние набранные очки')

    seasons = groups['opp'].values.flatten().tolist()
    pts = groups['pts'].values.flatten().tolist()
    plt.bar(seasons, pts)

    plt.grid('on')
    plt.savefig('static/mean_pts.png')
    return max_min_season_year['opp'].iloc[0]


def classification():
    """рассчет классификации"""
    x = stats[['pts', 'min']]
    y = stats['result']
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    print(clf.feature_names_in_)
    print(clf.feature_importances_)
    print(clf.score(x, y))
    names = ["Очки", "Минуты"]
    save_file(clf, "static/model.pkl")
    return names, clf.feature_importances_, clf.score(x, y),


def load_file(filename):
    return joblib.load(filename)


def save_file(model, filename):
    joblib.dump(model, filename)
