from search_engine import search_documents

ground_truth = {
    "movie about a ship sinking in the ocean": [
        "Titanic",
        "In the Heart of the Sea",
        "Life of Pi",
        "Cast Away",
        "The Perfect Storm",
        "Jaws",
        "Waterworld",
    ],
    "space survival mission": [
        "Interstellar",
        "Gravity",
        "The Martian",
        "Moon",
        "Sunshine",
        "Prometheus",
        "Apollo 13",
        "2001: A Space Odyssey",
        "Alien",
        "Aliens",
    ],
    "superhero saving the world": [
        "The Avengers",
        "Avengers: Age of Ultron",
        "The Dark Knight",
        "The Dark Knight Rises",
        "Batman Begins",
        "Man of Steel",
        "Spider-Man",
        "Spider-Man 2",
        "Iron Man",
        "Iron Man 2",
        "Iron Man 3",
        "Captain America: The First Avenger",
        "Captain America: The Winter Soldier",
        "Captain America: Civil War",
        "Thor",
        "Thor: The Dark World",
        "Ant-Man",
        "Deadpool",
        "X-Men",
        "X-Men: Days of Future Past",
        "Guardians of the Galaxy",
    ],
    "time travel adventure": [
        "Back to the Future",
        "Back to the Future Part II",
        "Back to the Future Part III",
        "Interstellar",
        "The Terminator",
        "Terminator 2: Judgment Day",
        "Looper",
        "Edge of Tomorrow",
        "About Time",
        "Midnight in Paris",
        "Source Code",
        "Donnie Darko",
    ],
    "artificial intelligence and robots": [
        "Ex Machina",
        "I, Robot",
        "The Terminator",
        "Terminator 2: Judgment Day",
        "A.I. Artificial Intelligence",
        "WALL·E",
        "Blade Runner",
        "Her",
        "Chappie",
        "The Matrix",
        "Transcendence",
        "Bicentennial Man",
        "Real Steel",
    ],
    "zombie apocalypse": [
        "World War Z",
        "28 Days Later",
        "28 Weeks Later",
        "Zombieland",
        "I Am Legend",
        "Dawn of the Dead",
        "Resident Evil",
        "Resident Evil: Apocalypse",
        "Resident Evil: Extinction",
        "Resident Evil: Afterlife",
        "Resident Evil: Retribution",
        "Warm Bodies",
        "Shaun of the Dead",
    ],
    "heist and robbery": [
        "Inception",
        "Ocean's Eleven",
        "Ocean's Twelve",
        "Ocean's Thirteen",
        "The Italian Job",
        "Now You See Me",
        "Now You See Me 2",
        "Inside Man",
        "The Town",
        "Snatch",
        "Lock, Stock and Two Smoking Barrels",
    ],
    "magical fantasy world": [
        "Harry Potter and the Philosopher's Stone",
        "Harry Potter and the Chamber of Secrets",
        "Harry Potter and the Prisoner of Azkaban",
        "Harry Potter and the Goblet of Fire",
        "Harry Potter and the Order of the Phoenix",
        "Harry Potter and the Half-Blood Prince",
        "The Lord of the Rings: The Fellowship of the Ring",
        "The Lord of the Rings: The Two Towers",
        "The Lord of the Rings: The Return of the King",
        "The Hobbit: An Unexpected Journey",
        "The Hobbit: The Desolation of Smaug",
        "The Hobbit: The Battle of the Five Armies",
        "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe",
        "Stardust",
        "Pan's Labyrinth",
        "Spirited Away",
        "Howl's Moving Castle",
        "Princess Mononoke",
    ],
    "war and military conflict": [
        "Saving Private Ryan",
        "Fury",
        "Black Hawk Down",
        "Apocalypse Now",
        "Platoon",
        "Full Metal Jacket",
        "Schindler's List",
        "Inglourious Basterds",
        "Dunkirk",
        "Pearl Harbor",
        "Hacksaw Ridge",
        "American Sniper",
        "Lone Survivor",
        "The Hurt Locker",
    ],
    "romantic comedy": [
        "The Proposal",
        "Notting Hill",
        "Bridget Jones's Diary",
        "How to Lose a Guy in 10 Days",
        "When Harry Met Sally...",
        "Pretty Woman",
        "Love Actually",
        "Crazy, Stupid, Love.",
        "Friends with Benefits",
        "No Strings Attached",
        "50 First Dates",
        "Hitch",
        "The Holiday",
        "27 Dresses",
        "Mamma Mia!",
    ],
}



def evaluate():
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for query, relevant_movies in ground_truth.items():
        results = search_documents(query)

        predicted = [r["Title"] for r in results]
        relevant = relevant_movies

        TP = len(set(predicted) & set(relevant))
        FP = len(set(predicted) - set(relevant))
        FN = len(set(relevant) - set(predicted))

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        print(f"\nQuery: {query}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" F1-score: {f1:.2f}")

    print("\n=== GLOBAL ===")
    print(f"Average Precision: {sum(all_precisions)/len(all_precisions):.2f}")
    print(f"Average Recall: {sum(all_recalls)/len(all_recalls):.2f}")
    print(f"Average F1-score: {sum(all_f1s)/len(all_f1s):.2f}")


if __name__ == "__main__":
    evaluate()
