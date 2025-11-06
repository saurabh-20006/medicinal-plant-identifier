import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ------------------------------
# 🌿 PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Indian Medicinal Plant Identifier 🌱",
    layout="wide",
    page_icon="🌿"
)


st.title("🌿 Indian Medicinal Plant & Leaf Identifier")
st.write("Upload an image of a plant or its leaf to identify and learn about it!")

# ------------------------------
# ⚙️ LOAD MODELS & CLASS NAMES
# ------------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

leaf_model = load_model("final_leaf_model.h5")
plant_model = load_model("final_plant_model.h5")

with open("class_names_leaf.json") as f:
    leaf_classes = json.load(f)
with open("class_names_plant.json") as f:
    plant_classes = json.load(f)

# ------------------------------
# 🔍 PREDICTION FUNCTION
# ------------------------------
def predict_image(model, image, class_names):
    # must match training size (128x128)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx]) * 100
    return class_names[idx], confidence

# ------------------------------
# 📸 IMAGE UPLOAD
# ------------------------------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        leaf_label, leaf_conf = predict_image(leaf_model, image, leaf_classes)
        plant_label, plant_conf = predict_image(plant_model, image, plant_classes)

    # ------------------------------
    # 📊 RESULTS
    # ------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌿 Leaf Prediction")
        st.success(f"**{leaf_label}** — {leaf_conf:.2f}% confidence")

    with col2:
        st.subheader("🌳 Plant Prediction")
        st.success(f"**{plant_label}** — {plant_conf:.2f}% confidence")

    # ------------------------------
    # 🧠 PLANT INFORMATION (basic)
    # ------------------------------
    plant_info = {
    "Aloevera": {
        "BENEFITS": "Heals burns, improves digestion, and nourishes skin and hair.",
        "FOUND_IN": "Tropical regions, commonly in Indian households.",
        "USES": "Used in gels, juices, cosmetics, and Ayurvedic medicines."
    },
    "Amla": {
        "BENEFITS": "Rich in Vitamin C, boosts immunity, improves hair health.",
        "FOUND_IN": "Found throughout India in deciduous forests.",
        "USES": "Used in chyawanprash, hair oils, and herbal tonics."
    },
    "Amruta_Balli": {
        "BENEFITS": "Boosts immunity and treats fever, cold, and diabetes.",
        "FOUND_IN": "Tropical India, especially southern regions.",
        "USES": "Used in Ayurvedic tonics and herbal teas."
    },
    "Arali": {
        "BENEFITS": "Used in treating skin diseases and ulcers.",
        "FOUND_IN": "Common in tropical and subtropical India.",
        "USES": "Used in Ayurvedic and Siddha medicines."
    },
    "Ashoka": {
        "BENEFITS": "Supports women's health, relieves menstrual pain.",
        "FOUND_IN": "Throughout India near forest areas.",
        "USES": "Used in herbal tonics and Ayurvedic syrups."
    },
    "Ashwagandha": {
        "BENEFITS": "Reduces stress, improves energy, and boosts stamina.",
        "FOUND_IN": "Dry regions of India.",
        "USES": "Used in powders, capsules, and rejuvenation tonics."
    },
    "Avacado": {
        "BENEFITS": "Rich in healthy fats, promotes heart and skin health.",
        "FOUND_IN": "Southern India, especially Kerala and Tamil Nadu.",
        "USES": "Used in salads, skincare, and smoothies."
    },
    "Bamboo": {
        "BENEFITS": "Improves joint health and boosts energy.",
        "FOUND_IN": "Common in Northeast and South India.",
        "USES": "Used in Ayurvedic formulations and construction."
    },
    "Basale": {
        "BENEFITS": "Heals wounds, reduces inflammation, and cools the body.",
        "FOUND_IN": "Tropical regions and home gardens.",
        "USES": "Used in traditional soups and leaf pastes."
    },
    "Betel": {
        "BENEFITS": "Improves digestion, freshens breath, and relieves cough.",
        "FOUND_IN": "Moist tropical areas across India.",
        "USES": "Used in paan and traditional medicine."
    },
    "Betel_Nut": {
        "BENEFITS": "Stimulates digestion and improves alertness.",
        "FOUND_IN": "Eastern and southern India.",
        "USES": "Used in chewing mixtures and Ayurveda."
    },
    "Brahmi": {
        "BENEFITS": "Improves memory, focus, and reduces anxiety.",
        "FOUND_IN": "Wetlands across India.",
        "USES": "Used in brain tonics and Ayurvedic formulations."
    },
    "Castor": {
        "BENEFITS": "Treats constipation and promotes hair growth.",
        "FOUND_IN": "Warm regions across India.",
        "USES": "Used in castor oil and Ayurvedic laxatives."
    },
    "Curry_Leaf": {
        "BENEFITS": "Rich in antioxidants, improves digestion, and hair health.",
        "FOUND_IN": "Southern India, cultivated in home gardens.",
        "USES": "Used in cooking and herbal oils."
    },
    "Doddapatre": {
        "BENEFITS": "Relieves cough, cold, and indigestion.",
        "FOUND_IN": "Western Ghats and South India.",
        "USES": "Used in herbal teas and syrups."
    },
    "Ekka": {
        "BENEFITS": "Treats fever, asthma, and skin diseases.",
        "FOUND_IN": "Dry areas of India.",
        "USES": "Used in traditional remedies and Ayurveda."
    },
    "Ganike": {
        "BENEFITS": "Used in treating diabetes and inflammation.",
        "FOUND_IN": "Common in rural India.",
        "USES": "Used in decoctions and home remedies."
    },
    "Gauva": {
        "BENEFITS": "Improves digestion and boosts immunity.",
        "FOUND_IN": "All parts of India.",
        "USES": "Used in juices, jams, and medicinal extracts."
    },
    "Geranium": {
        "BENEFITS": "Relieves stress and supports skin health.",
        "FOUND_IN": "Hill stations and gardens.",
        "USES": "Used in essential oils and aromatherapy."
    },
    "Henna": {
        "BENEFITS": "Cools the body and promotes scalp health.",
        "FOUND_IN": "North and West India.",
        "USES": "Used for hair dyeing and body art."
    },
    "Hibiscus": {
        "BENEFITS": "Promotes hair growth and controls cholesterol.",
        "FOUND_IN": "Grown widely in Indian gardens.",
        "USES": "Used in herbal hair oils and teas."
    },
    "Honge": {
        "BENEFITS": "Purifies air and used in traditional medicine.",
        "FOUND_IN": "Southern India, especially Karnataka.",
        "USES": "Used for biofuel and medicine."
    },
    "Insulin": {
        "BENEFITS": "Controls blood sugar levels naturally.",
        "FOUND_IN": "Grown in southern India.",
        "USES": "Used in diabetic treatments in Ayurveda."
    },
    "Jasmine": {
        "BENEFITS": "Reduces stress and promotes relaxation.",
        "FOUND_IN": "All over India, especially Tamil Nadu.",
        "USES": "Used in perfumes and aromatherapy."
    },
    "Lemon": {
        "BENEFITS": "Rich in Vitamin C, strengthens immunity, aids digestion.",
        "FOUND_IN": "Tropical and subtropical India.",
        "USES": "Used in beverages, skincare, and detox remedies."
    },
    "Lemon_grass": {
        "BENEFITS": "Relieves anxiety, boosts immunity, aids digestion.",
        "FOUND_IN": "South and Northeast India.",
        "USES": "Used in teas, oils, and aromatherapy."
    },
    "Mango": {
        "BENEFITS": "Boosts immunity, improves eyesight, and energy.",
        "FOUND_IN": "All over India, mainly Uttar Pradesh and Maharashtra.",
        "USES": "Used in juices, pickles, and medicines."
    },
    "Mint": {
        "BENEFITS": "Aids digestion and cools the body.",
        "FOUND_IN": "Widely cultivated across India.",
        "USES": "Used in teas, foods, and Ayurvedic syrups."
    },
    "Nagadali": {
        "BENEFITS": "Used in treating fever and inflammation.",
        "FOUND_IN": "Western and Southern India.",
        "USES": "Used in folk medicine and Ayurveda."
    },
    "Neem": {
        "BENEFITS": "Cleanses blood, treats skin issues, and fights bacteria.",
        "FOUND_IN": "Found throughout India.",
        "USES": "Used in soaps, oils, and herbal medicine."
    },
    "Nithyapushpa": {
        "BENEFITS": "Helps treat diabetes and joint pain.",
        "FOUND_IN": "Southern India.",
        "USES": "Used in Ayurvedic formulations."
    },
    "Nooni": {
        "BENEFITS": "Boosts immunity, improves skin, and relieves pain.",
        "FOUND_IN": "South India and coastal regions.",
        "USES": "Used in juices and traditional tonics."
    },
    "Pappaya": {
        "BENEFITS": "Aids digestion, promotes skin health, boosts immunity.",
        "FOUND_IN": "Tropical regions across India.",
        "USES": "Used in fruit extracts, masks, and herbal products."
    },
    "Pepper": {
        "BENEFITS": "Improves metabolism and relieves cough and cold.",
        "FOUND_IN": "Kerala, Karnataka, and Tamil Nadu.",
        "USES": "Used as spice and in Ayurvedic remedies."
    },
    "Pomegranate": {
        "BENEFITS": "Improves heart health and blood circulation.",
        "FOUND_IN": "Dry areas like Maharashtra and Gujarat.",
        "USES": "Used in juices, skincare, and herbal tonics."
    },
    "Raktachandini": {
        "BENEFITS": "Improves complexion and purifies blood.",
        "FOUND_IN": "South and Western India.",
        "USES": "Used in Ayurvedic beauty treatments."
    },
    "Rose": {
        "BENEFITS": "Cools the body, uplifts mood, and improves skin.",
        "FOUND_IN": "All across India, especially in Rajasthan.",
        "USES": "Used in rose water, perfumes, and Ayurvedic tonics."
    },
    "Sapota": {
        "BENEFITS": "Boosts energy and supports digestion.",
        "FOUND_IN": "Southern and Western India.",
        "USES": "Used in fruit desserts and tonics."
    },
    "Tulasi": {
        "BENEFITS": "Fights infections, purifies air, and boosts immunity.",
        "FOUND_IN": "Common in Indian households and temples.",
        "USES": "Used in teas, oils, and Ayurveda."
    },
    "Wood_sorel": {
        "BENEFITS": "Cools the body and aids digestion.",
        "FOUND_IN": "Moist and shaded regions across India.",
        "USES": "Used in herbal salads and folk medicine."
    }
}

    st.divider()
    st.subheader("📚 Plant Information")

    found_info = None
    for name, info in plant_info.items():
        if name.lower() in plant_label.lower() or name.lower() in leaf_label.lower():
            found_info = info
            st.success(f"🌿 **{name}**")
            st.write(f"**Benefits:** {info['BENEFITS']}")
            st.write(f"**Found In:** {info['FOUND_IN']}")
            st.write(f"**Uses:** {info['USES']}")
            break

    if not found_info:
        st.warning("ℹ️ No detailed information available for this plant. Try another image.")


# ------------------------------
# ✨ FOOTER
# ------------------------------
st.divider()
st.caption("Developed by **Saurabh (24156020)** ")
