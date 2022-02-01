import pickle
from pathlib import Path

import pandas as pd

# Label names taken from:
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/oxford_flowers102.py
_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

_SUBCLASS_COL_NAME = 'Class Name'
_SUPERCLASS_COL_NAME = 'Subclass'

SUPERCLASS_MAPPINGS_FILE_NAME = Path(__file__, '../data/superclass_mappings.pkl')

if __name__ == '__main__':
    # Create a mapping from label names to label ids for the subclass
    # The label id is the index + 1 of the label name
    subclass_name_to_id = {k: v + 1 for v, k in enumerate(_NAMES)}

    file_taxonomy = Path(__file__, '../taxonomy.csv')
    taxonomy = pd.read_csv(file_taxonomy)

    # Determine counts for each unique value
    x = taxonomy[_SUPERCLASS_COL_NAME].value_counts()
    print(len(x))
    print(x)

    # Create label ids for the superclasses
    superclass_names = taxonomy[_SUPERCLASS_COL_NAME].unique()
    superclass_name_to_id = {k: v + 1 for v, k in enumerate(superclass_names)}
    print(superclass_name_to_id)

    # Get the label id
    taxonomy_subclass_id = taxonomy[_SUBCLASS_COL_NAME].map(lambda x: subclass_name_to_id[x])
    taxonomy_superclass_id = taxonomy[_SUPERCLASS_COL_NAME].map(lambda x: superclass_name_to_id[x])

    # Create a mapping from the subclass label id to the superclass name
    superclasses_mappings = dict(zip(taxonomy_subclass_id, taxonomy_superclass_id))
    print(superclasses_mappings)

    # Save the mappings to a file
    with open(SUPERCLASS_MAPPINGS_FILE_NAME, 'wb') as f:
        pickle.dump(superclasses_mappings, f)
