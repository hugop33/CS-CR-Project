from utils.film_ontology import load_ontology
from config import DATA
from owlready2 import sync_reasoner_pellet


def get_related_films(onto, instance):
    related_films = []
    for prop in instance.get_properties():
        for value in prop[instance]:
            if isinstance(value, onto.Film):
                related_films.append(value)
    return related_films


def recommendation(onto, user_id):
    """
    Return the recommendations for a given user.
    """
    user = onto.User(user_id)
    requests = user.aCherché
    films = []
    counts = {}
    # for each request, add all the films that are linked by any relation to the request
    for request in requests:
        films += get_related_films(onto, request)
    # count the number of times each film appears
    for film in films:
        counts[film] = counts.get(film, 0) + 1
    # sort the films by the number of times they appear
    sorted_films = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_films


if __name__ == "__main__":
    onto = load_ontology(f'{DATA}smallIMDB_randusers.owl')
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
    recommendations = recommendation(onto, "u4")
    for f, c in recommendations:
        print(f"{f.aPourTitre} : {c}")

