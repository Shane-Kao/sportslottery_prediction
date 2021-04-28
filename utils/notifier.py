import requests

from configs import LINE_NOTIFY_TOKEN


def _notifier(msg):
    headers = {
        "Authorization": "Bearer " + LINE_NOTIFY_TOKEN,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    payload = {'message': msg}
    r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
    status_code = r.status_code
    if status_code != 200:
        raise Exception
    return status_code


if __name__ == "__main__":
    _notifier('\n123\n123')