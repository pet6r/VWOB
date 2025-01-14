#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <boost/asio/ssl/context.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <iostream>
#include <string>
#include <functional>
#include <thread>
#include <atomic>

namespace py = pybind11;
typedef websocketpp::client<websocketpp::config::asio_tls_client> client;

class WebSocketHandler {
public:
    WebSocketHandler() : is_connected_(false) {
        // Set logging level
        ws_client.clear_access_channels(websocketpp::log::alevel::all);
        ws_client.set_access_channels(websocketpp::log::alevel::connect);
        ws_client.set_access_channels(websocketpp::log::alevel::disconnect);
        ws_client.set_access_channels(websocketpp::log::alevel::app);

        ws_client.init_asio();

        // Enhanced TLS setup
        ws_client.set_tls_init_handler([](websocketpp::connection_hdl) {
            auto ctx = std::make_shared<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12);
            ctx->set_default_verify_paths();
            ctx->set_verify_mode(boost::asio::ssl::verify_peer);
            return ctx;
        });

        ws_client.set_message_handler([this](websocketpp::connection_hdl, client::message_ptr msg) {
            if (on_message_callback) {
                try {
                    on_message_callback(msg->get_payload());
                } catch (const std::exception& e) {
                    std::cerr << "Error in message callback: " << e.what() << std::endl;
                }
            }
        });

        ws_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
            is_connected_ = true;
            std::cout << "WebSocket connection established" << std::endl;

            if (!initial_sub_json.empty()) {
                websocketpp::lib::error_code ec;
                ws_client.send(hdl, initial_sub_json, websocketpp::frame::opcode::text, ec);
                if (ec) {
                    std::cerr << "Error sending subscription: " << ec.message() << std::endl;
                } else {
                    std::cout << "Subscription sent successfully" << std::endl;
                }
            }
        });

        ws_client.set_close_handler([this](websocketpp::connection_hdl) {
            is_connected_ = false;
            std::cout << "WebSocket connection closed" << std::endl;
        });

        ws_client.set_fail_handler([this](websocketpp::connection_hdl) {
            is_connected_ = false;
            std::cerr << "WebSocket connection failed" << std::endl;
        });
    }

    // Public member function declarations
    bool is_connected() const {
        return is_connected_;
    }

    bool reconnect() {
        if (is_connected_) {
            return true;
        }

        try {
            stop();
            connect(last_uri_);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Reconnection failed: " << e.what() << std::endl;
            return false;
        }
    }

    void connect(const std::string& uri) {
        try {
            connection_start_time_ = std::chrono::steady_clock::now();
            last_uri_ = uri;
            websocketpp::lib::error_code ec;
            auto con = ws_client.get_connection(uri, ec);

            if (ec) {
                std::cerr << "Error creating connection: " << ec.message() << std::endl;
                return;
            }

            m_hdl = con->get_handle();
            std::cout << "Connecting to: " << uri << std::endl;
            ws_client.connect(con);

            // Modify the open handler to include latency measurement
            ws_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
                is_connected_ = true;

                // Calculate connection establishment time
                auto now = std::chrono::steady_clock::now();
                auto connection_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - connection_start_time_).count();

                std::cout << "WebSocket connection established in " << connection_time << "ms" << std::endl;

                // Measure initial roundtrip latency
                measure_initial_latency();

                // Send subscription if available
                if (!initial_sub_json.empty()) {
                    websocketpp::lib::error_code ec;
                    ws_client.send(hdl, initial_sub_json, websocketpp::frame::opcode::text, ec);
                    if (ec) {
                        std::cerr << "Error sending subscription: " << ec.message() << std::endl;
                    } else {
                        std::cout << "Subscription sent successfully" << std::endl;
                    }
                }
            });

            ws_thread = std::thread([this]() {
                try {
                    ws_client.run();
                } catch (const std::exception& e) {
                    std::cerr << "WebSocket client exception: " << e.what() << std::endl;
                    is_connected_ = false;
                }
            });
        } catch (const std::exception& e) {
            std::cerr << "Exception during WebSocket connection: " << e.what() << std::endl;
            is_connected_ = false;
        }
    }

    bool ping() {
        if (!is_connected_) {
            return false;
        }

        try {
            auto start_time = std::chrono::steady_clock::now();
            auto hdl = m_hdl.lock();
            if (hdl) {
                websocketpp::lib::error_code ec;
                ws_client.ping(hdl, "", ec);
                if (!ec) {
                    auto end_time = std::chrono::steady_clock::now();
                    initial_latency_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time).count() / 1000.0;
                    return true;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Ping failed: " << e.what() << std::endl;
        }
        return false;
    }

    void stop() {
        try {
            ws_client.stop();
            is_connected_ = false;
            if (ws_thread.joinable()) {
                ws_thread.join();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during stop: " << e.what() << std::endl;
        }
    }

    void set_message_callback(std::function<void(const std::string&)> callback) {
        on_message_callback = callback;
    }

    void set_initial_subscription(const std::string& sub_json) {
        initial_sub_json = sub_json;
    }

    void send_message(const std::string& msg) {
        auto hdl = m_hdl.lock();
        if (hdl) {
            websocketpp::lib::error_code ec;
            ws_client.send(hdl, msg, websocketpp::frame::opcode::text, ec);
            if (ec) {
                std::cerr << "Error sending message: " << ec.message() << std::endl;
            }
        }
    }

    ~WebSocketHandler() {
        stop();
    }

    // Add these new methods
    double get_initial_latency() const {
        return initial_latency_ms_;
    }

    void measure_initial_latency() {
        if (!is_connected_) {
            std::cerr << "Cannot measure latency - not connected" << std::endl;
            return;
        }

        try {
            auto hdl = m_hdl.lock();
            if (hdl) {
                // Send a ping and measure roundtrip time
                auto start_time = std::chrono::steady_clock::now();

                websocketpp::lib::error_code ec;
                ws_client.ping(hdl, "latency_check", ec);

                if (ec) {
                    std::cerr << "Error sending ping: " << ec.message() << std::endl;
                    return;
                }

                // Add a pong handler specifically for latency measurement
                ws_client.set_pong_handler([this, start_time](websocketpp::connection_hdl, const std::string&) {
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                    initial_latency_ms_ = duration.count() / 1000.0;  // Convert to milliseconds

                    std::cout << "Initial connection latency: " << initial_latency_ms_ << "ms" << std::endl;
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "Error measuring latency: " << e.what() << std::endl;
        }
    }



private:
    client ws_client;
    websocketpp::connection_hdl m_hdl;
    std::function<void(const std::string&)> on_message_callback;
    std::thread ws_thread;
    std::string initial_sub_json;
    std::string last_uri_;
    std::atomic<bool> is_connected_;
    std::chrono::steady_clock::time_point connection_start_time_;
    double initial_latency_ms_ = 0.0;
};

PYBIND11_MODULE(websocket_handler, m) {
    py::class_<WebSocketHandler>(m, "WebSocketHandler")
        .def(py::init<>())
        .def("connect", &WebSocketHandler::connect)
        .def("stop", &WebSocketHandler::stop)
        .def("set_message_callback", &WebSocketHandler::set_message_callback)
        .def("set_initial_subscription", &WebSocketHandler::set_initial_subscription)
        .def("send_message", &WebSocketHandler::send_message)
        .def("is_connected", &WebSocketHandler::is_connected)
        .def("reconnect", &WebSocketHandler::reconnect)
        .def("ping", &WebSocketHandler::ping)
        .def("get_initial_latency", &WebSocketHandler::get_initial_latency);
}
