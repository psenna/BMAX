import thundersvm


class Svm:
    def grid_search(self, samples, labels):
        self.c = 1
        self.g = 0.1
        best_acc = 0
        for c in [10, 15, 30, 50, 80]:
            for g in [7, 10, 15, 20, 28, 35, 42, 50]:
                svm = thundersvm.SVC(gamma=g, C=c)
                svm.fit(samples, labels)
                acc = svm.score(samples, labels)

                print('Test SVM C = %4d, gamma = %2.2f, Acc = %.4f' % (c, g, acc))

                if acc > best_acc:
                    self.c = c
                    self.g = g
                    best_acc = acc

        return self.c, self.g

    def train(self, samples, labels, c=None, g=None):
        if c is None:
            if self.c is not None:
                c = self.c
            else:
                c = 100

        if g is None:
            if self.g is not None:
                g = self.g
            else:
                g = 0.5

        self.svm = thundersvm.SVC(gamma=g, C=c)
        self.svm.fit(samples, labels)
        acc = self.svm.score(samples, labels)
        print('##################################')
        print('SVM Trained with C = %d, gamma = %.2f, Acc = %.4f' % (c, g, acc))
        print('##################################')

    def test(self, samples, labels):
        acc = self.svm.score(samples, labels)
        print('##################################')
        print('SVM Tested Acc = %.4f' % acc)
        print('##################################')

    def predict(self, samples):
        return self.svm.predict(samples)

