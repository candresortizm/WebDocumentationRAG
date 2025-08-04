import os
import time
import uuid
import boto3
from boto3.dynamodb.conditions import Key
from pydantic import BaseModel, Field
from typing import List, Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

TABLA_HISTORIAL = os.environ.get("TABLA_HISTORIAL")

class HistorialModel(BaseModel):
    chat_id: Optional[str] = None
    create_time: int = Field(default_factory=lambda: int(time.time()))
    query_text: str
    answer_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    is_complete: bool = False

    @classmethod
    def get_table(cls: "HistorialModel") -> boto3.resource:
        dynamodb = boto3.resource("dynamodb")
        return dynamodb.Table(TABLA_HISTORIAL)

    def put_item(self):
        item = self.as_ddb_item()
        try:
            response = HistorialModel.get_table().put_item(Item=item)
            print(response)
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            raise e

    @classmethod
    def get_item(cls: "HistorialModel", chat_id: str) -> "HistorialModel":
        try:
            response = cls.get_table().query(KeyConditionExpression=Key('chat_id').eq(chat_id))
            print(response)
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            return None

        if "Items" in response and len(response["Items"])>0:
            item = response["Items"][0]
            return cls(**item)
        else:
            return None
    
    @classmethod
    def get_history(cls: "HistorialModel", chat_id: str) -> list["HistorialModel"]:
        try:
            response = cls.get_table().query(KeyConditionExpression=Key('chat_id').eq(chat_id))
            print(response)
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            return None

        if "Items" in response and len(response["Items"])>0:
            items = response["Items"]
            return items
        else:
            return None
        
    def update_item(cls: "HistorialModel") -> "HistorialModel":
        try:
            response = cls.get_table().update_item(
                                            Key={
                                                'chat_id': cls.chat_id,
                                                 'create_time': cls.create_time # Replace with the actual item's primary key
                                            },
                                            UpdateExpression="SET is_complete = :completeValue, answer_text = :ansValue",
                                            ExpressionAttributeValues={
                                                ':completeValue': 'true',
                                                ':ansValue': 'DOCUMENTOS CARGADOS'
                                            },
                                            ReturnValues="UPDATED_NEW" # Optional: return updated attributes
                                        )
        except ClientError as e:
            print("ClientError", e.response["Error"]["Message"])
            print(e)
            return None

        if "Item" in response:
            item = response["Item"]
            return cls(**item)
        else:
            return None
