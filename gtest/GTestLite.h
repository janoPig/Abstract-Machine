#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>
#include <iomanip>
#include <random>
#include <fstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace GTestLite {

#define CLR_GRN "\033[1;32m"
#define CLR_RED "\033[1;31m"
#define CLR_CYN "\033[1;36m"
#define CLR_RST "\033[0m"

    struct Failure { std::string file; int line; std::string message; };

    inline std::vector<Failure>& GetGlobalFailures() {
        static std::vector<Failure> global_failures;
        return global_failures;
    }

    inline thread_local std::vector<Failure>* g_current_test_failures = nullptr;

    inline std::vector<Failure>* GetActiveFailureVector() {
        if (g_current_test_failures != nullptr) {
            return g_current_test_failures;
        }
        return &GetGlobalFailures();
    }
    // -----------------------------------------

    struct TestResult {
        std::string suite;
        std::string name;
        long long micros;
        std::vector<Failure> failures;
    };

    class MessageProxy {
        std::vector<Failure>* failures;
        std::string file; int line; bool fatal;
        std::ostringstream oss;
    public:
        MessageProxy(std::vector<Failure>* f, std::string fl, int ln, std::string msg, bool fat)
            : failures(f), file(fl), line(ln), fatal(fat) {
            oss << msg;
        }
        ~MessageProxy() noexcept(false) {
            failures->push_back({ file, line, oss.str() });
            if (fatal) throw std::runtime_error("FATAL_TEST_FAILURE");
        }
        template<typename T> MessageProxy& operator<<(const T& val) { oss << "\n    Note: " << val; return *this; }
    };

    class Test {
    public:
        virtual ~Test() = default;
        virtual void SetUp() {}
        virtual void TearDown() {}
        virtual void Run() = 0;
        std::vector<Failure> failures;
    };

    struct Registry { std::string s, n; std::function<std::unique_ptr<Test>()> factory; };
    inline std::vector<Registry>& get_reg() { static std::vector<Registry> r; return r; }

    inline std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> char { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    inline std::string format_time(long long micros) {
        std::ostringstream oss; oss << std::fixed << std::setprecision(1);
        if (micros < 1000) oss << micros << " µs";
        else if (micros < 1000000) oss << (micros / 1000.0) << " ms";
        else oss << (micros / 1000000.0) << " s";
        return oss.str();
    }

    template <typename T, typename = void>
    struct has_to_string : std::false_type {};

    template <typename T>
    struct has_to_string<T, std::void_t<decltype(std::to_string(std::declval<T>()))>> : std::true_type {};

    template <typename T, typename = void>
    struct has_dump : std::false_type {};

    template <typename T>
    struct has_dump<T, std::void_t<decltype(std::declval<T>().Dump())>> : std::true_type {};

    template<typename T>
    std::string to_printable(const T& val)
    {
        if constexpr (std::is_convertible_v<T, std::string>)
        {
            return std::string(val);
        }
        else if constexpr (std::is_enum_v<T>)
        {
            return std::to_string(static_cast<long long>(val));
        }
        else if constexpr (has_to_string<T>::value)
        {
            return std::to_string(val);
        }
        else if constexpr (has_dump<T>::value)
        {
            return val.Dump();
        }
        else
        {
            return "object";
        }
    }

    inline int RunAllTests(int argc, char** argv) {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
        std::string filter = "", xml_path = "";
        int repeat = 1; unsigned int seed = (unsigned int)std::time(0);
        bool list = false, shuffle = false;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.find("--gtest_filter=") == 0) filter = arg.substr(15);
            if (arg.find("--gtest_repeat=") == 0) repeat = std::stoi(arg.substr(15));
            if (arg.find("--gtest_output=xml:") == 0) xml_path = arg.substr(19);
            if (arg.find("--gtest_random_seed=") == 0) seed = (unsigned int)std::stoi(arg.substr(20));
            if (arg == "--gtest_list_tests") list = true;
            if (arg == "--gtest_shuffle") shuffle = true;
        }

        auto& reg = get_reg();
        if (list) { for (auto& t : reg) std::cout << t.s << "." << t.n << "\n"; return 0; }
        if (shuffle) {
            std::shuffle(reg.begin(), reg.end(), std::default_random_engine(seed));
            std::cout << CLR_CYN << "[ SHUFFLE  ] " << CLR_RST << "Seed: " << seed << "\n";
        }

        std::vector<TestResult> results;
        int total_failed = 0;
        auto global_start = std::chrono::high_resolution_clock::now();

        for (int r = 0; r < repeat; r++) {
            for (auto& t : reg) {
                std::string full_name = t.s + "." + t.n;
                if (!filter.empty() && full_name.find(filter) == std::string::npos) continue;

                std::cout << CLR_GRN << "[ RUN      ] " << CLR_RST << full_name << "\n";
                auto inst = t.factory();
                g_current_test_failures = &inst->failures;

                auto start = std::chrono::high_resolution_clock::now();
                try {
                    inst->SetUp(); inst->Run(); inst->TearDown();
                }
                catch (const std::exception& e) {
                    if (std::string(e.what()) != "FATAL_TEST_FAILURE")
                        inst->failures.push_back({ __FILE__, __LINE__, "Unexpected exception: " + std::string(e.what()) });
                }
                catch (...) {
                    inst->failures.push_back({ __FILE__, __LINE__, "Unknown exception" });
                }
                auto end = std::chrono::high_resolution_clock::now();
                g_current_test_failures = nullptr;

                long long diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                results.push_back({ t.s, t.n, diff, inst->failures });

                if (inst->failures.empty()) {
                    std::cout << CLR_GRN << "[       OK ] " << CLR_RST << full_name << " (" << format_time(diff) << ")\n";
                }
                else {
                    for (auto& f : inst->failures)
                        std::cerr << CLR_RED << "[  FAILED  ] " << CLR_RST << f.file << ":" << f.line << ": " << f.message << "\n";
                    total_failed++;
                }
            }
        }

        auto global_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start).count() / 1000.0;

        if (!xml_path.empty()) {
            std::ofstream xml(xml_path);
            xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuites>\n"
                << "  <testsuite name=\"all\" tests=\"" << results.size() << "\" failures=\"" << total_failed << "\" time=\"" << total_s << "\">\n";
            for (const auto& res : results) {
                xml << "    <testcase name=\"" << res.name << "\" classname=\"" << res.suite << "\" time=\"" << (res.micros / 1000000.0) << "\">\n";
                for (const auto& f : res.failures) {
                    xml << "      <failure message=\"" << f.file << ":" << f.line << "\">" << f.message << "</failure>\n";
                }
                xml << "    </testcase>\n";
            }
            xml << "  </testsuite>\n</testsuites>\n";
        }

        std::cout << CLR_GRN << "[==========] " << CLR_RST << "Done. Total: " << std::fixed << std::setprecision(0) << (total_s * 1000) << " ms. Failed: " << (total_failed ? CLR_RED : CLR_GRN) << total_failed << CLR_RST << "\n";
        return total_failed > 0;
    }

} // namespace GTestLite

#define TEST(s, n) \
    class s##_##n##_T : public GTestLite::Test { public: void Run() override; }; \
    static inline int reg_##s##_##n = [](){ GTestLite::get_reg().push_back({#s, #n, [](){return std::make_unique<s##_##n##_T>();}}); return 0; }(); \
    void s##_##n##_T::Run()

#define TEST_F(f, n) \
    class f##_##n##_T : public f { public: void Run() override; }; \
    static inline int reg_##f##_##n = [](){ GTestLite::get_reg().push_back({#f, #n, [](){return std::make_unique<f##_##n##_T>();}}); return 0; }(); \
    void f##_##n##_T::Run()


#define GTEST_LITE_CHECK(a, b, op, fatal) \
    if (!((a) op (b))) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, \
        "Value of: " #a " " #op " " #b "\n    Actual: " + GTestLite::to_printable(a) + " vs " + GTestLite::to_printable(b), fatal)

#define EXPECT_EQ(a,b) GTEST_LITE_CHECK(a,b, ==, false)
#define ASSERT_EQ(a,b) GTEST_LITE_CHECK(a,b, ==, true)
#define EXPECT_NE(a,b) GTEST_LITE_CHECK(a,b, !=, false)
#define ASSERT_NE(a,b) GTEST_LITE_CHECK(a,b, !=, true)
#define EXPECT_LT(a,b) GTEST_LITE_CHECK(a,b, <,  false)
#define ASSERT_LT(a,b) GTEST_LITE_CHECK(a,b, <,  true)
#define EXPECT_GT(a,b) GTEST_LITE_CHECK(a,b, >,  false)
#define ASSERT_GT(a,b) GTEST_LITE_CHECK(a,b, >,  true)
#define EXPECT_LE(a,b) GTEST_LITE_CHECK(a,b, <=, false)
#define ASSERT_LE(a,b) GTEST_LITE_CHECK(a,b, <=, true)
#define EXPECT_GE(a,b) GTEST_LITE_CHECK(a,b, >=, false)
#define ASSERT_GE(a,b) GTEST_LITE_CHECK(a,b, >=, true)

#define EXPECT_TRUE(c)  if(!(c)) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected true: " #c, false)
#define ASSERT_TRUE(c)  if(!(c)) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected true: " #c, true)
#define EXPECT_FALSE(c) if((c))  GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected false: " #c, false)
#define ASSERT_FALSE(c) if((c))  GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected false: " #c, true)

#define EXPECT_STREQ(a,b)     { std::string s1=(a?a:""), s2=(b?b:""); if(s1!=s2) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected: " + s1 + " == " + s2, false); }
#define ASSERT_STREQ(a,b)     { std::string s1=(a?a:""), s2=(b?b:""); if(s1!=s2) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Expected: " + s1 + " == " + s2, true); }
#define EXPECT_STRCASEEQ(a,b) { if(GTestLite::to_lower(a?a:"") != GTestLite::to_lower(b?b:"")) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Case-insensitive match failed", false); }
#define ASSERT_STRCASEEQ(a,b) { if(GTestLite::to_lower(a?a:"") != GTestLite::to_lower(b?b:"")) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Case-insensitive match failed", true); }

#define EXPECT_THROW(stmt, ex) try { stmt; GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "No exception thrown", false); } catch(const ex&){} catch(...){ GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Wrong exception type", false); }
#define ASSERT_THROW(stmt, ex) try { stmt; GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "No exception thrown", true); } catch(const ex&){} catch(...){ GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Wrong exception type", true); }
#define EXPECT_NO_THROW(stmt)  try { stmt; } catch(...){ GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Unexpected exception thrown", false); }
#define ASSERT_NO_THROW(stmt)  try { stmt; } catch(...){ GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Unexpected exception thrown", true); }

#define EXPECT_NEAR(a,b,e)    if(std::abs((a)-(b)) > (e)) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Near check failed", false)
#define ASSERT_NEAR(a,b,e)    if(std::abs((a)-(b)) > (e)) GTestLite::MessageProxy(GTestLite::GetActiveFailureVector(), __FILE__, __LINE__, "Near check failed", true)
#define EXPECT_DOUBLE_EQ(a,b) EXPECT_NEAR(a,b, 1e-9)
#define ASSERT_DOUBLE_EQ(a,b) ASSERT_NEAR(a,b, 1e-9)

#define RUN_ALL_TESTS(argc, argv) GTestLite::RunAllTests(argc, argv)
