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
    for film_link in related_films:
        counts[film_link] = counts.get(film_link, 0) + 1

    return [[film,film_link,count] for film_link, count in counts.items() if count > thresh]


def recommendation(onto, user_id):

    """
    Return the recommendations for a given user.
    """

    user = onto.User(user_id)
    requests = user.aCherché
    likes = user.aLiké
    dislikes = user.aDisliké
    films_vus = []
    films_liked_near = []
    films_disliked_near = []
    films_search_near = []
    films = []
    score = {}
    coeff_exponential = 1.2

    # add film liked and disliked by the user in films_vus
    for film in likes+dislikes:
        films_vus.append(film)

    # add all the films that the user liked
        
    for film in likes:
        films_liked_near = [linked_film for linked_film in linked_films(onto, film) if linked_film[1] not in films_vus]
        for linked_film in films_liked_near:
            score[linked_film[1]] = score.get(linked_film[1], 0) + 3*linked_film[2]**coeff_exponential

    # add the films that the user disliked
    for film in dislikes:
        films_disliked_near = [linked_film for linked_film in linked_films(onto, film) if linked_film[1] not in films_vus]
        for linked_film in films_disliked_near:
            score[linked_film[1]] = score.get(linked_film[1], 0) - 3*linked_film[2]**coeff_exponential


    # for each request, add all the films that are linked by any relation to the request
    for request in requests:
        films_search_near = [film_related for film_related in get_related_films(onto, request) if film_related not in films_vus]
        for film_related in films_search_near:
            score[film_related] = score.get(film_related, 0) + 1

    
    # sort the films by the number of times they appear
    sorted_films = sorted(score.items(), key=lambda item: item[1], reverse=True)
    #afficher le nom des films déjà vus
    for film in films_vus:
        print(film.aPourTitre)
    return sorted_films
    

if __name__ == "__main__":
    onto = load_ontology(f'{DATA}smallIMDB_randusers_likes.owl')
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
    recommendations = recommendation(onto, "u4")
    for f, c in recommendations:
        print(f"{f.aPourTitre} : {c}")

