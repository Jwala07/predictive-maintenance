"""Send one message through Redpanda and read it back."""
import time

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

BROKER = "127.0.0.1:9092"
TOPIC = "smoke-test"

admin = AdminClient({"bootstrap.servers": BROKER})
admin.create_topics([NewTopic(TOPIC, num_partitions=1, replication_factor=1)])
time.sleep(2)

p = Producer({"bootstrap.servers": BROKER})
p.produce(TOPIC, key="m1", value="hello")
p.flush(10)

c = Consumer({"bootstrap.servers": BROKER, "group.id": "smoke",
              "auto.offset.reset": "earliest"})
c.subscribe([TOPIC])
for _ in range(20):
    msg = c.poll(1.0)
    if msg is not None and not msg.error():
        print("OK, received:", msg.value().decode())
        break
else:
    print("FAILED: no message received")
c.close()