import data, preprocessing

couples = data.get.couples_raw()
xy = data.get.couples_xy()
people = data.get.people_xy()[0]

normal = preprocessing.people.Standard()
erf = preprocessing.people.Erf()

print("----: normal")
print(normal.transform(people))

print("----: erf")
print(erf.transform(normal.transform(people)))

