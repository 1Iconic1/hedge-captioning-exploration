# Dedup few-shot
FEWSHOT_DEDUP_MESSAGES = [
    # Example 1: VizWiz_train_00000283.jpg
    {
        "role": "user",
        "content": """Atomic Statements:
The canned food has a yellow label.
The canned food has a green label.
The canned food has a red label.
The canned food is on a wooden surface.
The can contains spaghetti sauce.
The spaghetti sauce is traditional style.
The can is 15 ounces.
The can is 425 grams.
There is a small can.
The can has a picture on it.
The picture shows red pasta sauce.
The picture shows noodles.
The can is on a table.
The can is yellow.
The can is of national brand.
The can contains tomato paste.
The can is sitting on a counter.
The table is made of wood."""
    },
    {
        "role": "assistant",
        "content": """The canned food has a yellow label.
The canned food has a green label.
The canned food has a red label.
The canned food is on a wooden surface.
The picture shows red pasta sauce.
The spaghetti sauce is traditional style.
The can is 15 ounces.
The can is 425 grams.
The can has a picture on it.
The picture shows noodles.
The can is of national brand."""
    },

    # Example 2: VizWiz_train_00000196.jpg
    {
        "role": "user",
        "content": """Atomic Statements:
There is a red box.
The box is labeled Kaffe.
The box is next to a washing machine.
The package contains ground coffee.
The coffee is raspberry flavored.
The brand of the coffee is Gevalia Kaffe.
The box contains Gevalia coffee.
The coffee is Raspberry Danish flavored.
The box is on a table."""
    },
    {
        "role": "assistant",
        "content": """There is a red box.
The box is next to a washing machine.
The package contains ground coffee.
The coffee is raspberry flavored.
The brand of the coffee is Gevalia Kaffe.
The box is on a table."""
    },

    # Example 3: VizWiz_train_00000457.jpg
    {
        "role": "user",
        "content": """Atomic Statements:
There is a bottle.
The bottle contains shoe polish.
The shoe polish is for leather shoes.
The shoe polish is made by a company called Kiwi.
There is a container.
The container is of Kiwi leather shiner.
The container is being held up.
There is a product called KIWI brand shoe polish.
The shoe polish is for leather.
The shoe polish is labeled as 'premiere shine'.
The shoe polish is labeled as 'ultra brilliant'.
There is a liquid.
The liquid can be used for food."""
    },
    {
        "role": "assistant",
        "content": """There is a bottle.
The bottle contains shoe polish.
The shoe polish is for leather shoes.
The shoe polish is made by a company called Kiwi.
The container is being held up.
The shoe polish is labeled as 'premiere shine'.
The shoe polish is labeled as 'ultra brilliant'.
There is a liquid.
The liquid can be used for food."""
    },
]


# Recall few-shot
FEWSHOT_RECALL_MESSAGES = [
    # Example 1: VizWiz_train_00000283.jpg
    {
        "role": "user",
        "content": """Human-written atomic statements:
The canned food has a yellow label.
The canned food has a green label.
The canned food has a red label.
The canned food is on a wooden surface.
The can is 15 ounces.
The can is 425 grams.
The can contains spaghetti sauce.
The spaghetti sauce is traditional style.
The can is small.
The picture shows red pasta sauce.
The picture shows noodles.
The can is of national tomato paste.
The table is made of wood."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "The canned food is on a wooden surface.",
    "The can is 15 ounces.",
    "The can is 425 grams.",
    "The canned food has a yellow label.",
    "The canned food has a red label.",
    "The can is of national tomato paste."
  ],
  "FNs": [
    "The canned food has a green label.",
    "The can contains spaghetti sauce.",
    "The spaghetti sauce is traditional style.",
    "The can is small.",
    "The picture shows red pasta sauce.",
    "The picture shows noodles.",
    "The table is made of wood."
  ],
  "Match": [
    {"T_atomic": "The canned food is on a wooden surface.", "g_atomic": "The can is on a wooden surface."},
    {"T_atomic": "The can is 15 ounces.", "g_atomic": "The text reads 'WT. 15 OZ. (425g)'."},
    {"T_atomic": "The can is 425 grams.", "g_atomic": "The text reads 'WT. 15 OZ. (425g'."},
    {"T_atomic": "The canned food has a yellow label.", "g_atomic": "The label has a yellow color."},
    {"T_atomic": "The canned food has a red label.", "g_atomic": "The label has a red color."},
    {"T_atomic": "The can is of national tomato paste.", "g_atomic": "The text includes part of the word 'TIONAL'."}
  ],
  "Counts": {"TP": 6, "FN": 7}
}"""
    },

    # Example 2: VizWiz_train_00000196.jpg
    {
        "role": "user",
        "content": """Human-written atomic statements:
The box is red.
The box is placed next to a washing machine.
There is a washing machine.
The package is of Gevalia Kaffe brand.
The package contains ground coffee.
The coffee is raspberry flavored.
The flavor is Raspberry Danish.
The box is on a table.

Generated atomic statements:
- There is a red package.
- The package is labeled "GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee".
- There is a kitchen appliance.
- The kitchen appliance is black and white.
- The kitchen appliance is partially visible."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "The box is red.",
    "There is a washing machine.",
    "The package is of Gevalia Kaffe brand.",
    "The coffee is raspberry flavored.",
    "The flavor is Raspberry Danish."
  ],
  "FNs": [
    "The box is placed next to a washing machine.",
    "The package contains ground coffee.",
    "The box is on a table."
  ],
  "Match": [
    {"T_atomic": "The box is red.", "g_atomic": "There is a red package."},
    {"T_atomic": "There is a washing machine.", "g_atomic": "There is a kitchen appliance."},
    {"T_atomic": "The package is of Gevalia Kaffe brand.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."},
    {"T_atomic": "The coffee is raspberry flavored.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."},
    {"T_atomic": "The flavor is Raspberry Danish.", "g_atomic": "The package is labeled \"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\"."}
  ],
  "Counts": {"TP": 5, "FN": 3}
}"""
    },

    # Example 3: VizWiz_train_00000457.jpg
    {
        "role": "user",
        "content": """Human-written atomic statements:
There is a bottle.
The bottle contains shoe polish.
The shoe polish is for leather shoes.
The shoe polish is made by a company called Kiwi.
The container is being held up.
The shoe polish is labeled as cuir.
The shoe polish is labeled as premiere shine.
The shoe polish is labeled as ultra brilliant.
There is a liquid.
The liquid is used for food.

Generated atomic statements:
- A person is holding a container.
- The container is a Kiwi brand product.
- The product is for leather care.
- The product is designed for shoe care.
- The container is labeled "Premiere Shine Ultra-Brillant."
- The container is primarily red.
- The container is primarily black.
- There is a sponge applicator at the top of the container."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "The container is being held up.",
    "The shoe polish is labeled as premiere shine.",
    "The shoe polish is labeled as ultra brilliant."
  ],
  "FNs": [
    "There is a bottle.",
    "The bottle contains shoe polish.",
    "The shoe polish is for leather shoes.",
    "The shoe polish is made by a company called Kiwi.",
    "The shoe polish is labeled as cuir.",
    "There is a liquid.",
    "The liquid is used for food."
  ],
  "Match": [
    {"T_atomic": "The container is being held up.", "g_atomic": "A person is holding a container."},
    {"T_atomic": "The shoe polish is labeled as premiere shine.", "g_atomic": "The container is labeled \"Premiere Shine Ultra-Brillant.\""},
    {"T_atomic": "The shoe polish is labeled as ultra brilliant.", "g_atomic": "The container is labeled \"Premiere Shine Ultra-Brillant.\""}
  ],
  "Counts": {"TP": 3, "FN": 7}
}"""
    },
]


# Precision few-shot
FEWSHOT_PRECISION_MESSAGES = [
    # Example 1: VizWiz_train_00000283.jpg
    {
        "role": "user",
        "content": """Human-written captions:
- A canned food that has a yellow, green, and red label on a wooden surface.
- A 15 ounce (425g) can of traditional style spaghetti sauce.
- A small can with a picture of red pasta sauce on top of noodles on a table.
- A photo of a yellow can of national tomato paste sitting on a counter.
- A can with a yellow and red label sits on a wooden table.

Generated atomic statements:
- There is a can.
- The can is on a wooden surface.
- The can has visible text.
- The text reads 'WT. 15 OZ. (425g)'.
- The text includes part of the word 'TIONAL'.
- The label has a yellow color.
- The label has a red color."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "There is a can.",
    "The can is on a wooden surface.",
    "The text reads 'WT. 15 OZ. (425g)'",
    "The label has a yellow color.",
    "The label has a red color."
  ],
  "FPs": [
    "The text includes part of the word 'TIONAL'.",
    "The can has visible text."
  ],
  "Match": [
    {"g_atomic": "There is a can.", "T_org": "A can with a yellow and red label sits on a wooden table."},
    {"g_atomic": "The can is on a wooden surface.", "T_org": "A canned food that has a yellow, green, and red label on a wooden surface."},
    {"g_atomic": "The text reads 'WT. 15 OZ. (425g)'", "T_org": "A 15 ounce (425g) can of traditional style spaghetti sauce."},
    {"g_atomic": "The label has a yellow color.", "T_org": "A can with a yellow and red label sits on a wooden table."},
    {"g_atomic": "The label has a red color.", "T_org": "A can with a yellow and red label sits on a wooden table."}
  ],
  "Counts": {"TP": 6, "FP": 1}
}"""
    },

    # Example 2: VizWiz_train_00000196.jpg
    {
        "role": "user",
        "content": """Human-written captions:
- A red box of Kaffe placed next to a washing machine.
- A package of Gevalia Kaffe brand raspberry flavored ground coffee
- A box of Raspberry Danish flavored Gevalia coffee.
- Red box with raspberry flavored ground coffee on table.

Generated atomic statements:
- There is a red package.
- The package is labeled "GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee".
- There is a kitchen appliance.
- The kitchen appliance is black and white.
- The kitchen appliance is partially visible."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "There is a red package.",
    "The package is labeled \\"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\\".",
    "There is a kitchen appliance."
  ],
  "FPs": [
    "The kitchen appliance is black and white.",
    "The kitchen appliance is partially visible."
  ],
  "Match": [
    {"g_atomic": "There is a red package.", "T_org": "Red box with raspberry flavored ground coffee on table."},
    {"g_atomic": "The package is labeled \\"GEVALIA KAFFE Raspberry Danish Artificially Flavored Ground Coffee\\", "T_org": "A package of Gevalia Kaffe brand raspberry flavored ground coffee"},
    {"g_atomic": "There is a kitchen appliance.", "T_org": "A red box of Kaffe placed next to a washing machine."}
  ],
  "Counts": {"TP": 5, "FP": 0}
}"""
    },

    # Example 3: VizWiz_train_00000457.jpg
    {
        "role": "user",
        "content": """Human-written captions:
- I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi.
- A container of Kiwi leather shiner is being held up.
- KIWI brand shoe polish - leather cuir premiere shine ultra brilliant
- some type of liquid stuff you can use for food

Generated atomic statements:
- A person is holding a container.
- The container is a Kiwi brand product.
- The product is for leather care.
- The product is designed for shoe care.
- The container is labeled "Premiere Shine Ultra-Brillant."
- The container is primarily red.
- The container is primarily black.
- There is a sponge applicator at the top of the container."""
    },
    {
        "role": "assistant",
        "content": """{
  "TPs": [
    "A person is holding a container.",
    "The container is a Kiwi brand product.",
    "The product is for leather care.",
    "The product is designed for shoe care.",
    "The container is labeled \\"Premiere Shine Ultra-Brillant.\\""
  ],
  "FPs": [
    "The container is primarily red.",
    "The container is primarily black.",
    "There is a sponge applicator at the top of the container."
  ],
  "Match": [
    {"g_atomic": "A person is holding a container.", "T_org": "A container of Kiwi leather shiner is being held up."},
    {"g_atomic": "The container is a Kiwi brand product.", "T_org": "KIWI brand shoe polish - leather cuir premiere shine ultra brilliant"},
    {"g_atomic": "The product is for leather care.", "T_org": "I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi."},
    {"g_atomic": "The product is designed for shoe care.", "T_org": "I believe this is a bottle of leather shoe polish for premium shine made by a company called kiwi."},
    {"g_atomic": "The container is labeled \\"Premiere Shine Ultra-Brillant.\\"", "T_org": "KIWI brand shoe polish - leather cuir premiere shine ultra brilliant"}
  ],
  "Counts": {"TP": 5, "FP": 3}
}"""
    },
]
