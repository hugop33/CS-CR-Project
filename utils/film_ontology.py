import owlready2
from owlready2 import Thing, get_ontology, AllDisjoint, sync_reasoner_pellet, Imp, FunctionalProperty, ObjectProperty, owl_unionof
import json
import csv
import random

from config import OWLREADY2_JAVA_EXE, DATA

# Set the path to the Java executable
owlready2.JAVA_EXE = OWLREADY2_JAVA_EXE

DUMB_USER_NAMES = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10"]
USER_IDS = [f"u{i}" for i in range(1, 31)]


def parse_currency_to_int(currency_str):
    """
    Parse a currency string to an integer.
    Assumes the currency string is in a format like '$136,381,073' or '€136,381,073'.
    """
    # Remove currency symbol (if any) and commas
    cleaned_str = currency_str.lstrip('$€').replace(",", "")
    
    # Convert to integer
    try:
        return int(cleaned_str)
    except ValueError:
        return 0  # Return None or raise an error if conversion is not possible

def create_ontology():
    onto = get_ontology("http://www.semanticweb.org/ontologies/2020/1/IMDB.owl")
    return onto

def load_ontology(filename):
    onto = get_ontology(filename).load()
    return onto

def save_ontology(onto, filename):
    onto.save(file=filename, format="rdfxml")

def create_classes(onto):
    with onto:
        class User(Thing): pass
        class RequestItem(Thing): pass
        class Personne(RequestItem): pass
        class Film(RequestItem): pass
            
        class GenreFilm(RequestItem): pass

        class Pays(RequestItem): pass

        AllDisjoint([Personne, Film, GenreFilm, Pays, User])

        class Acteur(Personne): pass
        class Réalisateur(Personne): pass
        class Scénariste(Personne):  pass


        class aJouéDans(Acteur >> Film): pass
        class aPourDistribution(Film >> Acteur):
            inverse_property = aJouéDans

        class aRéalisé(Réalisateur >> Film): pass
        # réaliséPar est l'inverse de aRéalisé
        class réaliséPar(Film >> Réalisateur):
            inverse_property = aRéalisé

        class aÉcrit(Scénariste >> Film): pass
        class aPourScénariste(Film >> Scénariste):
            inverse_property = aÉcrit

        class aPourGenre(Film >> GenreFilm): pass
        class aPourPays(Film >> Pays): pass

        class noteIMDB(Film >> float, FunctionalProperty): pass
        class aPourTitre(Film >> str, FunctionalProperty): pass
        class aPourDate(Film >> int, FunctionalProperty): pass
        # The duration is a functional property
        class aPourDurée(Film >> int, FunctionalProperty): pass
        # The budget is a functional property
        class aPourBudget(Film >> int, FunctionalProperty): pass

        class aLiké(User >> Film): pass

        class LikéPar(Film >> User):
            inverse_property = aLiké
        
        class aDisliké(User >> Film): pass

        class DislikéPar(Film >> User):
            inverse_property = aDisliké

        class aCherché(User >> RequestItem): pass
        class CherchéPar(RequestItem >> User):
            inverse_property = aCherché


def load_movies_data(filename):
    movies_data = {}
    with open(filename, 'r') as file:
        movies = json.load(file)
        for movie in movies:
            imdb_id = movie.get('imdbID')
            if imdb_id:
                movies_data[imdb_id] = movie
    return movies_data

def populate_onto(onto, filenames, row_processors):
    assert len(filenames) == len(row_processors), "The number of filenames and row processors must be the same"
    for filename, row_processor in zip(filenames, row_processors):
        with open(filename, 'rt', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            count = 0
            for row in reader:
                if count > 0 and count%100000 == 0:
                    print(f'Processed {count} rows')
                row_processor(onto, row)
                count += 1

def process_title_basics(onto, movies_data, row):
    if row['titleType'] != 'movie' or row['isAdult'] != '0' or row['startYear'] < '1960':
        return
    movie_id = row['tconst']
    movie = onto.Film(name=f'movie_{movie_id}')
    movie.aPourTitre = row['primaryTitle']
    movie.aPourDate = int(row['startYear'])
    # Budget has to be looked for in movies_data.json
    if movies_data.get(movie_id):
        movie.aPourBudget = parse_currency_to_int(movies_data[movie_id].get('BoxOffice', 'N/A'))
    for genre in row['genres'].split(','):
        movie.aPourGenre.append(onto.GenreFilm(name=genre))

    # for country in row['country'].split(','):
    #     movie.aPourPays.append(Pays(name=country))
    
    # Add more fields as needed
    movie.aPourDurée = int(row['runtimeMinutes'])


def process_name_basics(onto, row):
    person_id = row['nconst']
    person = onto.Personne(name=f'person_{person_id}')
    person.nom = row['primaryName']
    person.profession = row['primaryProfession'].split(',')
    # Add more fields as needed
    person.knownForTitles = row['knownForTitles'].split(',')


def process_title_crew(onto, row):
    movie_id = row['tconst']
    movie = onto.Film(name=f'movie_{movie_id}')
    if movie:
        for director_id in row['directors'].split(','):
            if director_id != '\\N':
                director = onto.Personne(name=f'person_{director_id}')
                assert director
                director.hasRole = "Réalisateur"
                movie.réaliséPar.append(director)

        for writer_id in row['writers'].split(','):
            if writer_id != '\\N':
                writer = onto.Personne(name=f'person_{writer_id}')
                assert writer
                writer.hasRole = "Scénariste"
                movie.aPourScénariste.append(writer)

def process_title_principals(onto, row):
    movie_id = row['tconst']
    movie = onto.Film(name=f'movie_{movie_id}')
    if movie:
        principal = onto.Acteur(name=f'person_{row["nconst"]}')
        principal.hasRole = row['category']
        movie.aPourDistribution.append(principal)


def process_title_ratings(onto, row):
    movie_id = row['tconst']
    movie = onto.Film(name=f'movie_{movie_id}')
    if movie:
        movie.noteIMDB = float(row['averageRating'])


def film_population(onto):
    movies_data = load_movies_data(f'{DATA}movies_data.json')
    populate_onto(onto, 
        [f'{DATA}filtered_name_basics.tsv', f'{DATA}filtered_title_basics.tsv', f'{DATA}filtered_title_crew.tsv', f'{DATA}filtered_title_principals.tsv', f'{DATA}filtered_title_ratings.tsv'],
        [lambda onto, row: process_name_basics(onto, row), lambda onto, row: process_title_basics(onto, movies_data, row), lambda onto, row: process_title_crew(onto, row), lambda onto, row: process_title_principals(onto, row), lambda onto, row: process_title_ratings(onto, row)]
    )


def add_request(onto, user_id, attribute):
    """
    Add a request to the ontology.
    """
    with onto:
        # if user already in ontology, get it, else create it
        user = onto[user_id] if user_id in onto.User.instances() else onto.User(user_id)
        user.aCherché.append(attribute)

def add_like_dislike(onto, user_id, film, like=True):
    """
    Add a like or dislike to the ontology.
    """
    with onto:
        # if user already in ontology, get it, else create it
        user = onto[user_id] if user_id in onto.User.instances() else onto.User(user_id)
        if like:
            user.aLiké.append(film)
        else:
            user.aDisliké.append(film)


def random_requests_populating(onto, n_req, n_likes, n_dislikes):
    """
    Add n_req random requests to the ontology.
    Attributes requested by the users are chosen randomly.
    """
    users_req = [random.choice(USER_IDS) for _ in range(n_req)]
    users_likes = [random.choice(USER_IDS) for _ in range(n_likes)]
    users_dislikes = [random.choice(USER_IDS) for _ in range(n_dislikes)]
    film_instances = list(onto.Film.instances())
    acteur_instances = list(onto.Acteur.instances())
    réalisateur_instances = list(onto.Réalisateur.instances())
    genre_instances = list(onto.GenreFilm.instances())
    pays_instances = list(onto.Pays.instances())
    scénariste_instances = list(onto.Scénariste.instances())
    attributes = film_instances + acteur_instances + réalisateur_instances + genre_instances + pays_instances + scénariste_instances
    attributes_req = [random.choice(attributes) for _ in range(n_req)]
    attributes_likes = [random.choice(film_instances) for _ in range(n_likes)]
    attributes_dislikes = [random.choice(film_instances) for _ in range(n_dislikes)]
    for user_id, attribute in zip(users_req, attributes_req):
        add_request(onto, user_id, attribute)
    for user_id, film in zip(users_likes, attributes_likes):
        add_like_dislike(onto, user_id, film, like=True)
    for user_id, film in zip(users_dislikes, attributes_dislikes):
        add_like_dislike(onto, user_id, film, like=False)



if __name__ == "__main__":
    onto = create_ontology()
    create_classes(onto)
    film_population(onto)
    # sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    save_ontology(onto, f'{DATA}smallIMDB_test.owl')
    print("Ontology saved")


    onto_ = load_ontology(f'{DATA}smallIMDB_test.owl')
    
    random_requests_populating(onto_, 100, 50, 20)
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
    save_ontology(onto_, f'{DATA}smallIMDB_randusers_likes_test.owl')
    print("Ontology saved")

    # users_onto = onto

    users_onto = load_ontology(f'{DATA}smallIMDB_randusers_likes_test.owl')
    # get all requests from user
    uid = "u5"
    user = users_onto.User(uid)
    requests = user.aCherché
    print(f"User {uid} has requested {len(requests)} items")
    # for request in requests:
    #     # get all films linked to the request
    #     linked_films = request.