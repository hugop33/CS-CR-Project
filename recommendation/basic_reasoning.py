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

def linked_films(onto, film, thresh=2):
    """
    Return the films that are related to the given film.
    Return only the films that are related by more than thresh relations.
    """
    related_films = []
    for prop in film.get_properties():
        for value in prop[film]:
            try:
                value.get_properties()
            except AttributeError:
                continue
            for prop2 in value.get_properties():
                for value2 in prop2[value]:
                    if isinstance(value2, onto.Film):
                        related_films.append(value2)
    counts = {}
    for film in related_films:
        counts[film] = counts.get(film, 0) + 1
    return [film for film, count in counts.items() if count > thresh]


def recommendation(onto, user_id):
    """
    Return the recommendations for a given user.
    """
    user = onto.User(user_id)
    requests = user.aCherché
    likes = user.aLiké
    dislikes = user.aDisliké
    films_vus = []
    films = []
    counts = {}
    # for each request, add all the films that are linked by any relation to the request
    for request in requests:
        if isinstance(request, onto.Film):
            films_vus.append(request)
        films += get_related_films(onto, request)
    # add all the films that the user liked
    for film in likes:
        films_vus.append(film)
        films_liked_near = linked_films(onto, film)
        films += films_liked_near
    # remove the films that the user disliked
    for film in dislikes:
        films_vus.append(film)
        films_disliked_near = linked_films(onto, film)
        films += films_disliked_near

    # remove the films that the user has already seen
    films = [film for film in films if film not in films_vus]
    # count the number of times each film appears
    for film in films:
        counts[film] = counts.get(film, 0) + 1
        if film in films_liked_near:
            counts[film] += 3
        if film in films_disliked_near:
            counts[film] -= 3
    # sort the films by the number of times they appear
    sorted_films = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_films


if __name__ == "__main__":
    onto = load_ontology(f'{DATA}smallIMDB_randusers_likes.owl')
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
    recommendations = recommendation(onto, "u4")
    for f, c in recommendations:
        print(f"{f.aPourTitre} : {c}")

