import json, os

this_file = os.path.dirname(os.path.realpath(__file__))


def load_maps():
    path = os.path.join(this_file, "static", "map.json")
    print(path)
    return json.loads(open(path).read())


def make_link(name):
    split_name = name.split(" ")
    print(split_name)
    link = "<a href='/results?name=" + split_name[0] + "+" + split_name[1] + "'>" + name + "</a>"
    return link


def display(othername, index, l, people, name, people_in_relationships):
    if people[name]["gender"] == people[othername[0]]["gender"]:
        tr_class = ["same"]
    else:
        tr_class = ["diff"]

    tr_class.append(people[othername[0]]["group"])

    if othername[0] in people_in_relationships:
        color = "yellow"
    else:
        if index < l * .333:
            color = "green"
        elif index < l * .666:
            color = "black"
        else:
            color = "red"
    return '<tr class="' + " ".join(tr_class) + '"><td>' + str(index + 1) + '</td><td class="' + color + '">' + make_link(
        othername[0]) + '</td><td>' + str(
        round(othername[1], 2)) + "</td></tr>"


def make_block(maps, name):
    # we want to be able to show what place you are in for every other person
    places = []
    for person in maps["map"]:
        for i, othername in enumerate(maps["map"][person]):
            if othername[0] == name:
                places.append([person, i + 1])
                break
    places.sort(key=lambda x: x[1] - 1)

    output = "<br/><div class='block'>"

    output += "<h1>Results For: " + name + "</h1>"

    output += "<h3>You should go to Prom with:</h3>"
    try:
        message = maps["couples"][name]
    except KeyError:
        message = "Not enough people, sorry"
    output += "<h2>" + message + "</h2>"

    output += "<div class='group'><p>Where you are ranked for other people: </p> "
    output += "<table class='your-place'>"
    for i, person in enumerate(places):
        output += display(person, i, len(places), maps["people_raw"], name, maps["people_in_relationships"])
    output += "</table></div>"

    output += "<div class='group'><p>Best matches for you: </p>"
    output += "<table class='best-matches'>"
    for i, othername in enumerate(maps["map"][name]):
        output += display(othername, i, len(maps["map"][name]), maps["people_raw"], name, maps["people_in_relationships"])
    output += "</table></div>"

    output += "</div>"

    return output


def analyze(name):
    maps = load_maps()
    out = make_block(maps, name)
    return out


def typo(name):
    names = list(load_maps()["map"].keys())
    scored_names = []
    for othername in names:
        if othername[-1] == "2":
            continue
        score = levenshteinDistance(name, othername)
        scored_names.append([othername, score])
    scored_names.sort(key=lambda x: x[1])
    guess = scored_names[0][0].split(" ")
    print(guess)
    out = "<br/>Did you mean: <a href='/results?name=" + guess[0] + "+" + guess[1] + "'>" + guess[0] + " " + guess[
        1] + "</a>"
    print(out)
    return out


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def summary(name):
    """try:
        output = analyze(name)
    except KeyError:
        output = typo(name)"""
    output = analyze(name)
    return output
