from mongoengine import connect, Document, StringField, ListField, EmbeddedDocument, IntField, URLField, EmbeddedDocumentField

class SignInstanceEmbedded(EmbeddedDocument):
    gloss = StringField(required=True)
    video_url = URLField(required=True)
    filename = StringField(required=True)
    bbox = ListField(IntField())
    video_id = StringField(required=True)
    fps = IntField(required=True)

class SignInstance(Document):
    gloss = StringField(required=True)
    instances = ListField(EmbeddedDocumentField(SignInstanceEmbedded), required=True)