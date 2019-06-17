/*
2019.06.17 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the functions using template.
*/

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt, boost::property_tree::ptree::key_type const& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}