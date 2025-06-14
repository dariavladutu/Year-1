import requests
CLIENT_ID = "b9faac17926263f8534a"
CLIENT_SECRET = "ee118cdcba941562a49954fcccdab17d4bb1f2ff"
REDIRECT_URI = "https://httpbin.org/anything"

def create_oauth_link():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "user",
        "response_type": "code",
    }

    endpoint = "https://github.com/login/oauth/authorize"
    response = requests.get(endpoint, params=params)
    return response.url

#'https://github.com/login?client_id=b9faac17926263f8534a&return_to=%2Flogin%2Foauth%2Fauthorize%3Fclient_id%3Db9faac17926263f8534a%26redirect_uri%3Dhttps%253A%252F%252Fhttpbin.org%252Fanything%26response_type%3Dcode%26scope%3Duser'


def exchange_code_for_access_token(code=None):
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }

    headers = {"Accept": "application/json"}
    endpoint = "https://github.com/login/oauth/access_token"
    response = requests.post(endpoint, params=params, headers=headers).json()
    return response["access_token"]
#Exchanged code cbe71737a4107c3e1118 with access token: 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


def print_user_info(access_token=None):
    headers = {"Authorization": f"token {access_token}"}
    endpoint = "https://api.github.com/user"
    response = requests.get(endpoint, headers=headers).json()
    name = response["name"]
    username = response["login"]
    private_repos_count = response["total_private_repos"]
    print(
        f"{name} ({username}) | private repositories: {private_repos_count}"
    )

link = create_oauth_link()
print(f"Follow the link to start the authentication with GitHub: {link}")
code = input("GitHub code: ")
access_token = exchange_code_for_access_token(code)
print(f"Exchanged code {code} with access token: {access_token}")
print_user_info(access_token=access_token)

#None (dariavladutu236578) | private repositories: 0