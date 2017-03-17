import data, copy, vector_math

def val(*, people_xy_pre, couples_raw_pre, couples_xy_pre, alg_gen):
    people_raw = data.get.people_raw()
    people_xy = data.get.people_xy()
    for processor in people_xy_pre:
        people_xy[0] = processor.transform(people_xy[0])

    couples_raw = data.get.couples_raw()

    for alg in alg_gen():
        for i, couple in enumerate(couples_raw):
            new_couples = copy.deepcopy(couples_raw)
            del new_couples[i]

            for processor in couples_raw_pre:
                new_couples = processor.transform(new_couples)
            couples_xy = data.make.couples_xy(new_couples)
            for processor in couples_xy_pre:
                couples_xy = processor.transform(couples_xy)

            alg.fit(*couples_xy)
            soulmate_for_man = alg.predict_for_single_point(people_raw[couple["male"]])
            soulmate_for_woman = alg.predict_for_single_point(people_raw[couple["female"]])

            recs_for_man = vector_math.get_rec(people_raw, soulmate_for_man)
            recs_for_woman = vector_math.get_rec(people_raw, soulmate_for_woman)

            print(recs_for_man)
            input("")