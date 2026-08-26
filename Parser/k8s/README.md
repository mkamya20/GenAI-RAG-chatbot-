# gravity-spy-wiki-bot
## Setup
### Namespace
Run the following command to create the `gswiki-bot` namespace in kubernetes:
```bash
kubectl create namespace gswiki-bot
```
### Secrets
#### tls certs
this will use self-signed certs to encrypt traffic between app and reverse proxy:

```bash
# generate self-signed cert
openssl genrsa -out tls.key 4096
openssl req -new -x509 -key tls.key -out tls.crt -days 3650

# load tls certs into kubernetes secret
kubectl create secret tls -n gswiki-bot tls-certs --cert=./tls.crt --key=./tls.key
```

#### regcred
used to read images in gswiki-bot project in harbor.ischool.syr.edu:
```bash
kubectl create secret docker-registry regcred -n gswiki-bot --docker-server=harbor.ischool.syr.edu --docker-username='robot$gravity-spy+access' --docker-password=<password>
```

#### app secrets
create a file called `gravity-spy-wiki-bot-secret.yaml` and add the following:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gswiki-bot-secret
  namespace: gswiki-bot
stringData:
  aws-access-key-id: <string>
  aws-secret-access-key: <string>
  smtp-password: <string>
  azure-openai-api-key: <string>
  zot-api-key: <string>
```

then apply the secret to kubernetes:
```bash
kubectl apply -f ./gswiki-bot-secret.yaml
```
