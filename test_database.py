from database import init_database, get_all_news

init_database()

news = get_all_news()

print("--------------------------------")
print("PREVIOUS NEWS ARTICLES")
print("--------------------------------")

print("Total articles:", len(news))

for article in news[:5]:

    print("--------------------------------")

    print("ID:", article[0])

    print("User ID:", article[1])

    print("Title:", article[2])

    print("Language:", article[4])

    print("Date:", article[5])

    print("Article:", article[3][:100])