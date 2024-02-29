if __name__ == "__main__":
    from config import DATA, GENERATE_DB
    from recommendation.basic_reasoning import recommendation, load_ontology
    from owlready2 import sync_reasoner_pellet

    if GENERATE_DB:
        from utils.film_ontology import create_ontology, create_classes, film_population, save_ontology, random_requests_populating
        onto = create_ontology()
        create_classes(onto)
        film_population(onto)
        # sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
        save_ontology(onto, f'{DATA}smallIMDB.owl')
        print("Ontology saved")


        onto_ = load_ontology(f'{DATA}smallIMDB.owl')
        
        random_requests_populating(onto_, 100, 50, 20)
        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
        save_ontology(onto_, f'{DATA}smallIMDB_randusers_likes.owl')
        print("Ontology saved")

    onto = load_ontology(f'{DATA}smallIMDB_randusers_likes.owl')
    users_ids = [user.name for user in onto.User.instances()]
    chosen_user = input(f"Choose a user among {users_ids} : ")
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug=2)
    recommendations = recommendation(onto, chosen_user)
    for f, c in recommendations:
        print(f"{f.aPourTitre} : {c}")